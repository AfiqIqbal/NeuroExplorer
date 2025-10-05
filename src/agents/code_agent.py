"""Code agent placeholder supporting metadata reuse."""

from typing import Dict, List, Optional

from .base_agent import BaseAgent, FileAnalysis


class CodeAgent(BaseAgent):
    """Skeleton implementation for source code analysis with caching hooks."""

    def __init__(self) -> None:
        self._supported_formats = ['.py', '.js', '.ts', '.java', '.cs']

    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats

    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        metadata = metadata or {}
        # TODO: Integrate tree-sitter / CodeBERT to produce real summaries and embeddings.
        summary = metadata.get('summary', 'Code analysis not yet implemented.')
        tags = metadata.get('tags', ['code'])
        embedding = metadata.get('embedding', [])
        return FileAnalysis(summary=summary, tags=tags, embedding=embedding, metadata=metadata)
