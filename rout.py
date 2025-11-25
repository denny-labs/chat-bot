from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from src.services.chat_service import ChatService

app = FastAPI(title="SpaCy + sklearn Chatbot")


# -------- Request / Response Models --------
class ChatRequest(BaseModel):
    message: str
    sender: str = "user"


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[List[str]]
    reply: str
    sender: str


# -------- Initialize Chat Service (Singleton) --------
chat_service = ChatService()


# -------- CHAT ENDPOINT (ACCEPTS JSON + FORM DATA) --------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    message: Optional[str] = Form(None),
    sender: Optional[str] = Form("user")
):
    """
    Handles chat input.
    Supports both:
      • JSON body → { "message": "hi", "sender": "user" }
      • form-data → message=hi
    """

    # If message was not provided as form-data, fallback to JSON
    if message is None:
        try:
            body = await request.json()
            message = body.get("message", None)
            sender = body.get("sender", "user")
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message must be a non-empty string.")

    # Process message via ChatService
    response = await chat_service.process_message(message, sender)
    return response
