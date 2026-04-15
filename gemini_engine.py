import os
import json
import httpx
import asyncio
import re
import pexpect
import platform
from typing import AsyncGenerator

# Fallback sequence yang sah: 2.0 Flash -> 1.5 Flash -> 1.5 Flash-8B -> 1.5 Pro
MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro"
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_INSTRUCTION_BASE = """You are Gemini-Final, a high-level expert AI.
You can execute code using pexpect, analyze the device environment, and generate images when requested.
Current Device Environment:
{env_info}
"""

TOOLS_DEFINITION = [
    {
        "function_declarations": [
            {
                "name": "generate_image",
                "description": "Generate an image based on a descriptive prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Detailed description"}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "execute_shell",
                "description": "Execute a shell command using pexpect.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command"},
                        "input_text": {"type": "string", "description": "Optional input"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
]

def get_env_info():
    return json.dumps({
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "termux": "TERMUX_VERSION" in os.environ,
        "shell": os.environ.get("SHELL"),
        "cwd": os.getcwd()
    }, indent=2)

async def chat_stream(message: str, history: list = None) -> AsyncGenerator[dict, None]:
    if history is None: history = []
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield {"type": "error", "content": "GEMINI_API_KEY not found"}
        return

    current_model_idx = 0
    env_info = get_env_info()
    
    while current_model_idx < len(MODELS):
        model_name = MODELS[current_model_idx]
        url = f"{GEMINI_API_BASE}/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
        
        payload = {
            "contents": history + [{"role": "user", "parts": [{"text": message}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION_BASE.format(env_info=env_info)}]},
            "tools": TOOLS_DEFINITION,
            "generationConfig": {"temperature": 0.7}
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 429 or response.status_code == 404:
                        current_model_idx += 1
                        continue
                    
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield {"type": "error", "content": f"Error {response.status_code}: {err_text.decode()}"}
                        break

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "): continue
                        try:
                            chunk = json.loads(line[6:])
                            candidates = chunk.get("candidates", [])
                            if not candidates: continue
                            
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield {"type": "text", "content": part["text"]}
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    yield {"type": "tool", "name": fc["name"], "args": fc["args"]}
                        except: continue
                    break
        except Exception as e:
            current_model_idx += 1
            if current_model_idx >= len(MODELS):
                yield {"type": "error", "content": str(e)}

async def run_pexpect(command: str, input_text: str = None):
    try:
        child = pexpect.spawn(command, encoding='utf-8')
        if input_text:
            child.sendline(input_text)
        child.expect(pexpect.EOF)
        return child.before
    except Exception as e:
        return str(e)
