import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.agents import registry

async def test_pdf_agent():
    """Test the PDF agent with a sample PDF file."""
    # Get the PDF agent
    pdf_agent = registry.agents.get('PDFAgent')
    if not pdf_agent:
        print("PDF agent not found in registry!")
        print("Available agents:", list(registry.agents.keys()))
        return
    
    # Path to the sample PDF
    pdf_path = os.path.join('test_samples', 'sample.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"Test PDF not found at {pdf_path}")
        return
    
    print(f"Testing PDF agent with: {pdf_path}")
    
    try:
        # Analyze the PDF
        result = await pdf_agent.analyze(pdf_path)
        
        # Print results
        print("\n=== PDF Analysis Results ===")
        print(f"Summary: {result.summary}")
        print(f"Tags: {', '.join(result.tags)}")
        print(f"Embedding size: {len(result.embedding)} dimensions")
        
        if result.metadata:
            print("\n=== Extracted Text (first 200 chars) ===")
            content = result.metadata.get('content', '')
            print(content[:200] + '...' if content else 'No content')
            
            print("\n=== Metadata ===")
            for key, value in result.metadata.items():
                if key != 'content':
                    value_str = str(value)
                    print(f"{key}: {value_str if len(value_str) < 100 else value_str[:100] + '...'}")
    except Exception as e:
        print(f"Error analyzing PDF: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_pdf_agent())
