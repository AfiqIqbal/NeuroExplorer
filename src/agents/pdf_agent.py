import os
import hashlib
from typing import Dict, List, Optional, Any
from PyPDF2 import PdfReader
from .base_agent import BaseAgent, FileAnalysis

class PDFAgent(BaseAgent):
    """Agent for processing PDF files."""
    
    @property
    def supported_formats(self) -> List[str]:
        """List of file extensions this agent can handle."""
        return ['.pdf']
    
    def __init__(self):
        self.agent_name = "pdf_agent"
        self.display_name = "PDF Document"
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                metadata = reader.metadata or {}
                
                # Convert metadata to a dictionary
                meta_dict = {}
                for key, value in metadata.items():
                    if hasattr(value, 'encode'):
                        meta_dict[key.lower().replace('/', '_')] = str(value)
                
                # Add basic file info
                meta_dict['page_count'] = len(reader.pages)
                
                return meta_dict
        except Exception as e:
            print(f"Warning: Could not extract PDF metadata: {str(e)}")
            return {}
    
    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        """Analyze a PDF file and return its metadata and content."""
        if metadata is None:
            metadata = {}
        
        # Extract text content (synchronous operation)
        content = self.extract_text(file_path)
        
        # Extract metadata (synchronous operation)
        pdf_metadata = self.extract_metadata(file_path)
        
        # Generate a summary (first 500 chars for now, could be enhanced with AI)
        summary = content[:500].strip() + ('...' if len(content) > 500 else '')
        
        # Generate tags (simple for now, could be enhanced with NLP)
        tags = ['pdf']
        if 'title' in pdf_metadata:
            tags.append(pdf_metadata['title'].lower().replace(' ', '-'))
            
        # Generate embedding (placeholder - would use a proper embedding model in production)
        # In a real implementation, this would be an async operation
        embedding = [0.0] * 384  # Example: 384-dimension zero vector
        
        # Add file stats to metadata
        file_stat = os.stat(file_path)
        metadata.update({
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'file_size': file_stat.st_size,
            'created': file_stat.st_ctime,
            'modified': file_stat.st_mtime,
            **pdf_metadata,
            'file_type': 'pdf',
            'agent': self.agent_name,
            'agent_version': '1.0.0',
            'content': content,
        })
        
        return FileAnalysis(
            summary=summary,
            tags=tags,
            embedding=embedding,
            metadata=metadata
        )

# The agent will be registered in __init__.py
