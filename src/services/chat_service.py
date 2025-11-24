from typing import Dict, Any, List, Tuple
from src.services.agent_service import AgentService


class ChatService:
    def __init__(self):
        self.agent = AgentService()

    async def process_message(self, message: str, sender: str) -> Dict[str, Any]:
        # Get intent and confidence
        intent, confidence = await self.agent.predict_intent(message)

        # Get entities
        entities = await self.agent.extract_entities(message)
        # convert to simple list format for JSON (list of [text, label])
        entities_list: List[List[str]] = [[e[0], e[1]] for e in entities]

        # Get reply text
        reply = await self.agent.get_reply(intent)

        return {
            "intent": intent,
            "confidence": float(confidence),
            "entities": entities_list,
            "reply": reply,
            "sender": sender
        }
