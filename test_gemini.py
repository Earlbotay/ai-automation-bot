import asyncio
import os
from gemini_engine import chat_stream

async def test():
    # Set API Key secara manual untuk ujian jika belum ada di env
    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "AIzaSyCMcwp6LAJMiiGRXkr4IHtkagtnn0-xpxw"
    
    print("Testing Gemini Connection...")
    async for chunk in chat_stream("Hai"):
        print(f"Received chunk: {chunk}")

if __name__ == "__main__":
    asyncio.run(test())
