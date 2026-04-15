import asyncio
import os
from gemini_engine import chat_stream

async def test():
    # Pastikan API Key sudah diset di environment variable (GitHub Secrets)
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY tidak ditemukan di environment!")
        return
    
    print("Testing Gemini Connection...")
    async for chunk in chat_stream("Hai"):
        print(f"Received chunk: {chunk}")

if __name__ == "__main__":
    asyncio.run(test())
