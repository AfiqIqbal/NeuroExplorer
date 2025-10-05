import asyncio
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..agents import registry
from ..agents.base_agent import BaseAgent, FileAnalysis

IGNORED_DIRECTORIES = {
    '.git',
    '.venv',
    '__pycache__',
    '.mypy_cache',
    'node_modules',
}

class FileOrchestrator:
    """Orchestrates file scanning and processing."""

    def __init__(self, registry=registry, existing_records: Optional[Dict[str, Dict]] = None):
        self.registry = registry
        self.processed_files: Set[str] = set()
        self.existing_records = existing_records or {}
        
    async def scan_directory(self, directory: str, recursive: bool = True) -> List[Dict]:
        """
        Scan a directory for files and analyze them.
        
        Args:
            directory: Path to the directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of analysis results for processed files
        """
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
            
        tasks = []
        processed = []
        
        for root, dirs, files in os.walk(directory):
            # Prune ignored directories in-place so os.walk() skips them
            dirs[:] = [
                d for d in dirs
                if d not in IGNORED_DIRECTORIES and not d.startswith('.')
            ]
            for file in files:
                file_path = os.path.join(root, file)
                if self._should_process(file_path):
                    tasks.append(self.process_file(file_path))
                    
            if not recursive:
                break
                
        # Process files concurrently
        if tasks:
            processed = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Filter out any exceptions
        return [r for r in processed if isinstance(r, dict)]
    
    async def process_file(self, file_path: str) -> Optional[Dict]:
        """Process a single file with the appropriate agent."""
        try:
            agent = self.registry.get_agent_for_file(file_path)
            if not agent:
                logging.debug(f"No agent found for file: {file_path}")
                return None
                
            # Skip if already processed (check with normalized path)
            abs_path = os.path.abspath(file_path)
            if abs_path in self.processed_files:
                return None

            existing = self.existing_records.get(abs_path)
            file_hash = self._compute_file_hash(file_path)

            if existing and file_hash and existing.get('content_hash') == file_hash:
                logging.info(f"Reusing cached analysis for: {file_path}")
                cached_metadata = existing.get('metadata') or {}
                if file_hash and 'content_hash' not in cached_metadata:
                    cached_metadata = {**cached_metadata, 'content_hash': file_hash}
                analysis = FileAnalysis(
                    summary=existing.get('summary', ''),
                    tags=existing.get('tags', []),
                    embedding=existing.get('embedding', []),
                    metadata=cached_metadata,
                )
                self.processed_files.add(abs_path)

                return {
                    'path': file_path,
                    'analysis': analysis,
                    'agent': existing.get('agent') or agent.__class__.__name__,
                    'content_hash': file_hash,
                    'reused': True,
                }

            logging.info(f"Processing file: {file_path}")
            analysis = await agent.analyze(file_path, metadata=existing.get('metadata') if existing else None)
            if analysis.metadata is None:
                analysis.metadata = {}
            if file_hash:
                analysis.metadata.setdefault('content_hash', file_hash)
            self.processed_files.add(abs_path)

            # Update in-memory cache for subsequent files in this run
            self.existing_records[abs_path] = {
                'summary': analysis.summary,
                'tags': analysis.tags,
                'embedding': analysis.embedding,
                'metadata': analysis.metadata,
                'agent': agent.__class__.__name__,
                'content_hash': file_hash,
                'mtime': os.path.getmtime(file_path) if os.path.exists(file_path) else None,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else None,
            }
            
            return {
                'path': file_path,
                'analysis': analysis,
                'agent': agent.__class__.__name__,
                'content_hash': file_hash,
                'reused': False,
            }
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _should_process(self, file_path: str) -> bool:
        """Determine if a file should be processed."""
        path = Path(file_path)

        if path.name.startswith('.'):
            return False

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            return False

        agent = self.registry.get_agent_for_file(file_path)
        if not agent:
            return False

        abs_path = os.path.abspath(file_path)
        record = self.existing_records.get(abs_path)
        if record and record.get('content_hash'):
            try:
                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)
            except OSError:
                return False

            stored_mtime = record.get('mtime')
            stored_size = record.get('file_size')
            if (
                stored_mtime is not None
                and stored_size is not None
                and math.isclose(stored_mtime, current_mtime, rel_tol=0, abs_tol=1e-6)
                and stored_size == current_size
            ):
                logging.debug(f"Skipping unchanged file: {file_path}")
                return False

        return True

    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> Optional[str]:
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError as exc:
            logging.error(f"Failed to hash {file_path}: {exc}")
            return None
