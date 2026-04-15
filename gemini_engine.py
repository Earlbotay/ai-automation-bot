import os
import json
import httpx
import asyncio
import re
import pexpect
import platform
from typing import AsyncGenerator

# Fallback sequence: 3.0 Flash -> 3.0 Lite -> 2.5 Flash -> 2.5 Lite
MODELS = [
    "gemini-3.0-flash",
    "gemini-3.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = """You are Gemini-Final, a high-level expert AI.
You can execute code using pexpect, analyze the device environment, and generate images when requested.
Current Device Environment:
{env_info}
"""

TOOLS_DEFINITION = [
    {
        "function_declarations": [
            {
                "name": "generate_image",
                "description": "Generate an image based on a descriptive prompt. Trigger this when the user explicitly asks for an image, drawing, or picture.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image to generate"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "execute_shell",
                "description": "Execute a shell command using pexpect with background input support.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run"
                        },
                        "input_text": {
                            "type": "string",
                            "description": "Optional input to send to the command"
                        }
                    },
                    "required": ["command"]
                }
            }
        ]
    }
]

def get_env_info():
    info = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "termux": "TERMUX_VERSION" in os.environ,
        "shell": os.environ.get("SHELL"),
        "cwd": os.getcwd()
    }
    return json.dumps(info, indent=2)

async def chat_stream(message: str, history: list = None) -> AsyncGenerator[dict, None]:
    if history is None: history = []
    
    current_model_idx = 0
    env_info = get_env_info()
    
    while current_model_idx < len(MODELS):
        model_name = MODELS[current_model_idx]
        url = f"{GEMINI_API_BASE}/models/{model_name}:streamGenerateContent?alt=sse&key={API_KEY}"
        
        payload = {
            "contents": history + [{"role": "user", "parts": [{"text": message}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION.format(env_info=env_info)}]},
            "tools": TOOLS_DEFINITION,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 429: # Rate limit
                        current_model_idx += 1
                        continue
                    
                    if response.status_code != 200:
                        yield {"type": "error", "content": f"Error {response.status_code}"}
                        break

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "): continue
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
                                # Auto-execute tool logic here or in sub-agent
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
