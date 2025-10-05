from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FileAnalysis:
    """Container for file analysis results."""
    summary: str
    tags: List[str]
    embedding: List[float]  # Vector embedding of the file content
    metadata: Dict = None

class BaseAgent(ABC):
    """Base class for all file type agents."""
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of file extensions this agent can handle (e.g., ['.txt', '.md'])."""
        pass
    
    @abstractmethod
    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        """
        Analyze the file and return its content analysis.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileAnalysis object containing the analysis results
        """
        pass
    
    def can_handle(self, file_path: str) -> bool:
        """Check if this agent can handle the given file."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_formats)
