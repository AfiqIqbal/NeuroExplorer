from typing import List, Dict, Optional
import logging
import os
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
            
            # Read file content
            try:
                text = self._read_file(file_path)
                if not text or not text.strip():
                    return FileAnalysis(
                        summary="Empty file",
                        tags=["empty"],
                        embedding=[],
                        metadata={
                            'file_path': file_path,
                            'file_name': os.path.basename(file_path),
                            'file_type': 'text/plain',
                            'error': 'File is empty'
                        }
                    )
            except Exception as e:
                return FileAnalysis(
                    summary=f"Error reading file: {str(e)}",
                    tags=["error"],
                    embedding=[],
                    metadata={
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'file_type': 'text/plain',
                        'error': str(e)
                    }
                )
            
            # Add file stats to metadata
            file_stat = os.stat(file_path)
            metadata.update({
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'file_size': file_stat.st_size,
                'created': file_stat.st_ctime,
                'modified': file_stat.st_mtime,
                'file_type': 'text/plain',
                'agent': 'text_agent',
                'agent_version': '1.0.0',
                'content': text,
                'char_count': len(text),
                'word_count': len(text.split()),
                'line_count': len(text.splitlines())
            })
            
            # Generate summary (first 200 chars or first few lines)
            summary = self._generate_summary(text)
            
            # Extract tags (simple word frequency based for now)
            tags = self._extract_tags(text)
            
            # Generate embedding (using the first 1000 chars to avoid very large inputs)
            try:
                embedding = self.model.encode([text[:1000]])[0].tolist()
            except Exception as e:
                print(f"Warning: Could not generate embedding: {str(e)}")
                embedding = [0.0] * 384  # Fallback to zero vector
            
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
                tags=["error"],
                embedding=[]
            )
    
    def _read_file(self, file_path: str) -> str:
        """Read file content based on file extension."""
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.pdf':
                # Let the PDF agent handle PDF files
                from .pdf_agent import PDFAgent
                pdf_agent = PDFAgent()
                return pdf_agent.extract_text(file_path)
            elif ext == '.docx':
                return docx2txt.process(file_path)
            else:  # .txt, .md, etc.
                # Try different encodings if utf-8 fails
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                # If all encodings fail, try binary read as last resort
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='replace')
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text."""
        if not text.strip():
            return "No content"
            
        # Try to find the first paragraph or meaningful block
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            # Use the first paragraph if it's not too long
            first_paragraph = paragraphs[0].strip()
            if len(first_paragraph) <= max_length * 1.5:  # Allow some flexibility
                return first_paragraph
        
        # Otherwise, take the first few sentences or first 200 chars
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        if sentences:
            summary = sentences[0]
            for s in sentences[1:]:
                if len(summary) + len(s) + 1 <= max_length:  # +1 for the period
                    summary += '. ' + s
                else:
                    break
            return summary + ('.' if not summary.endswith('.') else '')
        
        # Fallback to simple truncation
        return text[:max_length].strip() + ('...' if len(text) > max_length else '') + '.'
    
    def _extract_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Extract tags from text."""
        import re
        from collections import Counter
        
        # Common words to exclude
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 
            'have', 'was', 'were', 'this', 'that', 'with', 'from', 'your', 'will',
            'has', 'had', 'they', 'their', 'what', 'which', 'when', 'where', 'who'
        }
        
        # Extract words (alphanumeric with at least 3 chars)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Count word frequencies, excluding stop words
        word_count = Counter(
            word for word in words 
            if word not in stop_words and not word.isdigit()
        )
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        word_counts = Counter(filtered_words)
        
        # Get most common words as tags
        return [word for word, _ in word_counts.most_common(max_tags)]

