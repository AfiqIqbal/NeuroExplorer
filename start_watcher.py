#!/usr/bin/env python3
"""
Starts the file watcher service to monitor the test_samples directory.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('file_watcher.log')
    ]
)

from src.services.file_watcher import start_file_watcher

async def main():
    # Get the directory to watch (default to test_samples in the project root)
    watch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_samples'))
    
    print(f"Starting file watcher for directory: {watch_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        await start_file_watcher(watch_dir, recursive=True)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.error(f"Error in file watcher: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
