from typing import List, Dict, Optional
import logging
from pathlib import Path
from ..agents.base_agent import BaseAgent, FileAnalysis
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx2txt

class TextAgent(BaseAgent):
    """Agent for processing text files, PDFs, and Word documents."""
    
    def __init__(self):
        self._supported_formats = ['.txt', '.pdf', '.docx', '.md']
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats
        
    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        """Analyze a text file and return its content analysis."""
        try:
            metadata = metadata or {}
            # Read file content based on type
            text = self._read_file(file_path)
            if not text.strip():
                return FileAnalysis(
                    summary="Empty file",
                    tags=["empty"],
                    embedding=[]
                )
                
            # Generate summary (simplified - could be enhanced with LLM)
            summary = self._generate_summary(text)
            
            # Extract keywords/tags (simplified - could use better NLP)
            tags = self._extract_tags(text)
            
            # Generate embedding
            embedding = self.model.encode([text])[0].tolist()
            
            return FileAnalysis(
                summary=summary,
                tags=tags,
                embedding=embedding,
                metadata={
                    **metadata,
                    'char_count': len(text),
                    'word_count': len(text.split()),
                    'line_count': len(text.splitlines())
                }
            )
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {str(e)}")
            return FileAnalysis(
                summary=f"Error analyzing file: {str(e)}",
                tags=["error"],
                embedding=[]
            )
    
    def _read_file(self, file_path: str) -> str:
        """Read content from different file types."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self._read_pdf(file_path)
        elif ext == '.docx':
            return docx2txt.process(file_path)
        else:  # .txt, .md, etc.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or '')
        return '\n'.join(text)
    
    def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the text."""
        # This is a simple implementation - could be enhanced with better NLP
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return '. '.join(sentences[:max_sentences]) + '.'
    
    def _extract_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Extract important keywords/tags from text."""
        # Simple implementation - could be enhanced with better NLP
        from collections import Counter
        import re
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Common English stop words to exclude
        stop_words = set([
            'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 
            'with', 'as', 'by', 'this', 'that', 'is', 'are', 'was', 'were'
        ])
        
        # Filter and count words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        word_counts = Counter(filtered_words)
        
        # Get most common words as tags
        return [word for word, _ in word_counts.most_common(max_tags)]

