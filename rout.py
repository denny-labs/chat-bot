from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

from src.services.chat_service import ChatService

app = FastAPI(title="SpaCy + sklearn Chatbot")

# Request / Response models
class ChatRequest(BaseModel):
    message: str
    sender: str = "user"

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[List[str]]
    reply: str
    sender: str

# Single ChatService instance (manages loading & models internally)
chat_service = ChatService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message must be a non-empty string.")

    response = await chat_service.process_message(req.message, req.sender)
    return response
