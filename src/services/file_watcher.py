"""
File watcher service that monitors a directory for changes and processes files using the appropriate agent.
"""
import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.agents import registry
from src.database.models import FileAnalysis, db

logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles file processing with the appropriate agent."""
    
    def __init__(self):
        self.processing_files: Set[str] = set()
        
    async def process_file(self, file_path: str) -> Optional[FileAnalysis]:
        """Process a file using the appropriate agent and store results in the database."""
        try:
            file_path = os.path.abspath(file_path)
            
            # Skip if already processing this file
            if file_path in self.processing_files:
                return None
                
            self.processing_files.add(file_path)
            
            # Get file stats
            try:
                stat = os.stat(file_path)
                file_size = stat.st_size
                mtime = stat.st_mtime
            except OSError as e:
                logger.error(f"Error getting file stats for {file_path}: {str(e)}")
                return None
            
            # Calculate content hash (for change detection)
            content_hash = self._calculate_file_hash(file_path)
            
            # Check if file has been processed and not modified
            with db.get_session() as session:
                existing = session.query(FileAnalysis).filter_by(file_path=file_path).first()
                if existing and existing.mtime == mtime and existing.content_hash == content_hash:
                    logger.debug(f"Skipping unchanged file: {file_path}")
                    return existing
            
            # Get the appropriate agent
            agent = registry.get_agent_for_file(file_path)
            if not agent:
                logger.warning(f"No agent found for file: {file_path}")
                return None
                
            logger.info(f"Processing {file_path} with {agent.__class__.__name__}")
            
            # Process the file
            try:
                analysis = await agent.analyze(file_path)
                if not analysis:
                    logger.error(f"No analysis returned for {file_path}")
                    return None
                
                # Create or update database record
                file_analysis = FileAnalysis.from_analysis(
                    file_path=file_path,
                    analysis=analysis,
                    file_type=Path(file_path).suffix.lower(),
                    file_size=file_size,
                    mtime=mtime,
                    agent_name=getattr(agent, 'agent_name', None),
                    content_hash=content_hash
                )
                
                with db.get_session() as session:
                    # Check if record exists
                    existing = session.query(FileAnalysis).filter_by(file_path=file_path).first()
                    if existing:
                        # Update existing record
                        for key, value in file_analysis.__dict__.items():
                            if not key.startswith('_') and key != 'id':
                                setattr(existing, key, value)
                        session.commit()
                        logger.info(f"Updated analysis for {file_path}")
                        return existing
                    else:
                        # Add new record
                        session.add(file_analysis)
                        session.commit()
                        logger.info(f"Added new analysis for {file_path}")
                        return file_analysis
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
                return None
                
        finally:
            self.processing_files.discard(file_path)
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""


class FileWatcher(FileSystemEventHandler):
    """Watches for file system events and triggers processing."""
    
    def __init__(self, watch_dir: str, processor: FileProcessor):
        self.watch_dir = os.path.abspath(watch_dir)
        self.processor = processor
        self.pending_tasks: Dict[str, asyncio.Task] = {}
        
    def on_created(self, event):
        """Called when a file or directory is created."""
        if not event.is_directory:
            self._schedule_processing(event.src_path)
    
    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if not event.is_directory:
            self._schedule_processing(event.src_path)
    
    def on_moved(self, event):
        """Called when a file or directory is moved/renamed."""
        if not event.is_directory:
            # Process the new file path
            self._schedule_processing(event.dest_path)
    
    def _schedule_processing(self, file_path: str):
        """Schedule a file for processing with debounce."""
        file_path = os.path.abspath(file_path)
        
        # Cancel any pending task for this file
        if file_path in self.pending_tasks:
            self.pending_tasks[file_path].cancel()
        
        # Schedule a new task with a small delay to handle rapid changes
        async def process_with_delay():
            try:
                await asyncio.sleep(1)  # Debounce delay
                await self.processor.process_file(file_path)
            except asyncio.CancelledError:
                pass  # Task was cancelled, which is expected
            finally:
                if file_path in self.pending_tasks:
                    del self.pending_tasks[file_path]
        
        self.pending_tasks[file_path] = asyncio.create_task(process_with_delay())


async def start_file_watcher(watch_dir: str, recursive: bool = True):
    """Start the file watcher service."""
    # Ensure watch directory exists
    watch_dir = os.path.abspath(watch_dir)
    os.makedirs(watch_dir, exist_ok=True)
    
    # Initialize database
    db.init_db()
    
    # Create processor and watcher
    processor = FileProcessor()
    event_handler = FileWatcher(watch_dir, processor)
    
    # Set up the observer
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=recursive)
    observer.start()
    
    logger.info(f"Watching directory: {watch_dir}")
    
    try:
        # Initial processing of existing files
        await process_existing_files(watch_dir, processor)
        
        # Keep the watcher running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
    finally:
        observer.stop()
        observer.join()
        
        # Cancel any pending tasks
        for task in event_handler.pending_tasks.values():
            task.cancel()
        
        # Wait for all tasks to complete
        if event_handler.pending_tasks:
            await asyncio.wait(list(event_handler.pending_tasks.values()))


async def process_existing_files(directory: str, processor: FileProcessor):
    """Process all existing files in the directory."""
    logger.info(f"Processing existing files in {directory}...")
    
    # Process files in parallel with a semaphore to limit concurrency
    sem = asyncio.Semaphore(5)  # Limit to 5 concurrent processes
    
    async def process_file(file_path: str):
        async with sem:
            await processor.process_file(file_path)
    
    # Collect all files
    files_to_process = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if registry.get_agent_for_file(file_path):
                files_to_process.append(file_path)
    
    # Process files in batches
    batch_size = 10
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i + batch_size]
        await asyncio.gather(*(process_file(f) for f in batch))
    
    logger.info(f"Finished processing {len(files_to_process)} files")
