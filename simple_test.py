import asyncio
import os
from src.agents.text_agent import TextAgent

async def main():
    # Create a text agent
    agent = TextAgent()
    
    # Path to the license file
    file_path = os.path.join('test_samples', 'LICENSE.txt')
    
    # Test the agent
    try:
        print(f"Testing with file: {file_path}")
        result = await agent.analyze(file_path)
        print("\n=== Analysis Results ===")
        print(f"Summary: {result.summary}")
        print(f"Tags: {result.tags}")
        print(f"Embedding size: {len(result.embedding)}")
        
        if result.metadata:
            print("\n=== Metadata ===")
            for key, value in result.metadata.items():
                if key != 'content':
                    print(f"{key}: {value}" if len(str(value)) < 100 else f"{key}: {str(value)[:100]}...")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
