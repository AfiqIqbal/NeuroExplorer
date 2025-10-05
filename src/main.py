import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from .core.orchestrator import FileOrchestrator
from .database.models import db, FileAnalysis as DBAnalysis

# Initialize database
db.init_db()


def load_existing_records() -> dict[str, dict]:
    session = db.get_session()
    records: dict[str, dict] = {}
    try:
        for row in session.query(DBAnalysis).all():
            extra = row.extra_metadata or {}
            records[os.path.abspath(row.file_path)] = {
                'summary': row.summary or '',
                'tags': row.tags or [],
                'embedding': row.embedding or [],
                'metadata': extra,
                'agent': extra.get('agent'),
                'content_hash': row.content_hash or extra.get('content_hash'),
                'mtime': row.mtime,
                'file_size': row.file_size,
            }
    finally:
        session.close()
    return records

async def main():
    """Main entry point for the NeuroExplorer application."""
    parser = argparse.ArgumentParser(description='NeuroExplorer - Context-aware file explorer')
    parser.add_argument('directory', type=str, help='Directory to scan')
    parser.add_argument('--recursive', action='store_true', help='Scan directories recursively')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    existing_records = load_existing_records()

    # Initialize the orchestrator
    orchestrator = FileOrchestrator(existing_records=existing_records)
    
    try:
        # Scan the directory and process files
        logger.info(f"Scanning directory: {args.directory}")
        results = await orchestrator.scan_directory(args.directory, recursive=args.recursive)
        
        # Save results to database
        session = db.get_session()
        try:
            for result in results:
                # Check if file already exists in DB
                existing = session.query(DBAnalysis).filter_by(file_path=result['path']).first()
                file_size = os.path.getsize(result['path']) if os.path.exists(result['path']) else None
                mtime = os.path.getmtime(result['path']) if os.path.exists(result['path']) else None
                content_hash = result.get('content_hash')
                
                if existing:
                    # Update existing record
                    existing.summary = result['analysis'].summary
                    existing.tags = result['analysis'].tags
                    existing.embedding = result['analysis'].embedding
                    metadata = result['analysis'].metadata or {}
                    metadata['agent'] = result['agent']
                    if content_hash:
                        metadata.setdefault('content_hash', content_hash)
                    existing.extra_metadata = metadata
                    existing.content_hash = content_hash
                    existing.file_size = file_size
                    existing.mtime = mtime
                else:
                    # Create new record
                    db_analysis = DBAnalysis.from_analysis(
                        file_path=result['path'],
                        analysis=result['analysis'],
                        file_type=Path(result['path']).suffix.lower(),
                        agent_name=result['agent'],
                        content_hash=content_hash,
                        file_size=file_size,
                        mtime=mtime,
                    )
                    session.add(db_analysis)
                
                action = "reused" if result.get('reused') else "processed"
                logger.info(f"{action.title()}: {result['path']} ({result['agent']})")
                
            session.commit()
            logger.info(f"Processed {len(results)} files")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=args.debug)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
