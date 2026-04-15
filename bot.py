import os
import asyncio
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from gemini_engine import chat_stream, run_pexpect

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

class SubAgent:
    def __init__(self, model_fallback_logic):
        self.logic = model_fallback_logic

    async def do_task(self, task_description):
        # Sub-agent uses the same engine logic
        response = ""
        async for chunk in chat_stream(f"SUB-AGENT TASK: {task_description}"):
            if chunk["type"] == "text":
                response += chunk["content"]
        return response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gemini-Final Ready. System autonomous.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID and OWNER_ID != 0:
        return

    user_msg = update.message.text
    full_reply = ""
    reply_msg = await update.message.reply_text("Thinking...")

    history = [] # In production, load from cache/db
    
    async for chunk in chat_stream(user_msg, history):
        if chunk["type"] == "text":
            full_reply += chunk["content"]
            if len(full_reply) % 50 == 0: # Update UI periodically
                await reply_msg.edit_text(full_reply + " 🧠")
        
        elif chunk["type"] == "tool":
            tool_name = chunk["name"]
            args = chunk["args"]
            
            if tool_name == "execute_shell":
                await reply_msg.edit_text(f"Executing: {args['command']}...")
                result = await run_pexpect(args["command"], args.get("input_text"))
                # Feed result back to AI
                async for sub_chunk in chat_stream(f"SHELL RESULT: {result}", history + [{"role": "user", "parts": [{"text": user_msg}]}]):
                    if sub_chunk["type"] == "text":
                        full_reply += sub_chunk["content"]
                        await reply_msg.edit_text(full_reply)

            elif tool_name == "generate_image":
                await reply_msg.edit_text(f"Generating image: {args['prompt']}...")
                # Placeholder for Imagen integration (use existing logic if available)
                await update.message.reply_text(f"🎨 Image request detected: {args['prompt']}")

    if full_reply:
        await reply_msg.edit_text(full_reply)

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
