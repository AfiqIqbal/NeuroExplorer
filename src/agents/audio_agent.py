"""Audio agent placeholder supporting metadata reuse."""

from typing import Dict, List, Optional

from .base_agent import BaseAgent, FileAnalysis


class AudioAgent(BaseAgent):
    """Skeleton implementation for audio analysis with caching hooks."""

    def __init__(self) -> None:
        self._supported_formats = ['.wav', '.mp3', '.flac']

    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats

    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        metadata = metadata or {}
        # TODO: Integrate Whisper + YAMNet and populate summary/tags/embedding.
        summary = metadata.get('summary', 'Audio analysis not yet implemented.')
        tags = metadata.get('tags', ['audio'])
        embedding = metadata.get('embedding', [])
        return FileAnalysis(summary=summary, tags=tags, embedding=embedding, metadata=metadata)
