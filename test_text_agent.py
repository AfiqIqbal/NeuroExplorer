import os
import asyncio
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

from src.agents import registry

async def test_text_agent():
    """Test the Text agent with a sample text file."""
    # Get the text agent
    text_agent = registry.agents.get('TextAgent')
    if not text_agent:
        print("Text agent not found in registry!")
        print("Available agents:", list(registry.agents.keys()))
        return
    
    # Path to the sample text file
    text_path = os.path.join('test_samples', 'LICENSE.txt')
    
    if not os.path.exists(text_path):
        print(f"Test text file not found at {text_path}")
        return
    
    print(f"Testing Text agent with: {text_path}")
    
    try:
        # Analyze the text file
        result = await text_agent.analyze(text_path)
        
        # Print results
        print("\n=== Text Analysis Results ===")
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
        print(f"Error analyzing text file: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_text_agent())
