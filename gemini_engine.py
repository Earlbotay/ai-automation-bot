import os
import json
import httpx
import asyncio
import re
import pexpect
import platform
from typing import AsyncGenerator

# Fallback sequence yang anda minta: 3.0 Flash -> 3.0 Lite -> 2.5 Flash -> 2.5 Lite
MODELS = [
    "gemini-3.0-flash",
    "gemini-3.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_INSTRUCTION_BASE = """You are Gemini-Final, a high-level expert AI by Earlstore.
You execute code via pexpect and provide deep analytical responses.
Device: {env_info}
"""

TOOLS_DEFINITION = [
    {
        "function_declarations": [
            {
                "name": "generate_image",
                "description": "Triggered when user wants an image or visual.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"]
                }
            },
            {
                "name": "execute_shell",
                "description": "Execute shell command via pexpect.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "input_text": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
]

def get_env_info():
    return json.dumps({
        "platform": platform.platform(),
        "node": platform.node(),
        "termux": "TERMUX_VERSION" in os.environ,
        "cwd": os.getcwd()
    })

async def chat_stream(message: str, history: list = None) -> AsyncGenerator[dict, None]:
    if history is None: history = []
    api_key = os.getenv("GEMINI_API_KEY")
    env_info = get_env_info()
    
    current_model_idx = 0
    while current_model_idx < len(MODELS):
        model_name = MODELS[current_model_idx]
        url = f"{GEMINI_API_BASE}/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
        
        payload = {
            "contents": history + [{"role": "user", "parts": [{"text": message}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION_BASE.format(env_info=env_info)}]},
            "tools": TOOLS_DEFINITION,
            "generationConfig": {"temperature": 0.9}
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=payload) as response:
                    # Jika model tidak wujud (404) atau rate limit (429), cuba model seterusnya
                    if response.status_code in [404, 429, 400]:
                        current_model_idx += 1
                        continue
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "): continue
                        try:
                            chunk = json.loads(line[6:])
                            parts = chunk.get("candidates", [])[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part: yield {"type": "text", "content": part["text"]}
                                elif "functionCall" in part: yield {"type": "tool", "name": part["functionCall"]["name"], "args": part["functionCall"]["args"]}
                        except: continue
                    break
        except:
            current_model_idx += 1
            if current_model_idx >= len(MODELS): yield {"type": "error", "content": "All models failed."}

async def run_pexpect(command: str, input_text: str = None):
    try:
        child = pexpect.spawn(command, encoding='utf-8', timeout=60)
        if input_text: child.sendline(input_text)
        child.expect(pexpect.EOF)
        return child.before
    except Exception as e: return str(e)
