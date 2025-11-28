# from typing import Dict, Any, List, Tuple
# from src.services.agent_service import AgentService


# class ChatService:
#     def __init__(self):
#         self.agent = AgentService()

#     async def process_message(self, message: str, sender: str) -> Dict[str, Any]:
#         # Get intent and confidence
#         intent, confidence = await self.agent.predict_intent(message)

#         # Get entities
#         entities = await self.agent.extract_entities(message)
#         # convert to simple list format for JSON (list of [text, label])
#         entities_list: List[List[str]] = [[e[0], e[1]] for e in entities]

#         # Get reply text
#         reply = await self.agent.get_reply(intent)

#         return {
#             "intent": intent,
#             "confidence": float(confidence),
#             "entities": entities_list,
#             "reply": reply,
#             "sender": sender
#         }

import json
from pathlib import Path
import random

from src.nlu_initializer import NLUInitializer
from src.nlu_pipeline import NLUPipeline


class ChatService:
    def __init__(self, intents_path: str = "data/intents.json"):
        self.intents_path = Path(intents_path)

        # Load intents.json
        self.intents = self._load_intents()

        # Load all NLU components
        nlu_init = NLUInitializer()
        models = nlu_init.load_all()

        self.nlu = NLUPipeline(
            intent_model=models["intent_model"],
            vectorizer=models["vectorizer"],
            spacy_nlp=models["spacy_nlp"]
        )

    def _load_intents(self):
        if not self.intents_path.exists():
            raise FileNotFoundError("Cannot load data/intents.json")

        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("intents", [])

    def _get_response_for_intent(self, intent_name: str) -> str:
        """
        Returns a random response string for the given intent.
        """
        for item in self.intents:
            if item["intent"] == intent_name:
                return random.choice(item.get("responses", ["I have a response but it's empty."]))
        return "I'm not sure how to respond to that."

    async def process_message(self, message: str, sender: str = "user") -> dict:
        """
        Main bot handler.
        Performs:
         - Intent detection
         - Entity extraction
         - Template selection
        """

        # Run NLU
        nlu_result = self.nlu.run(message)

        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]
        entities = nlu_result["entities"]

        # Get reply template
        reply = self._get_response_for_intent(intent)

        # Build output JSON
        return {
            "sender": "bot",
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "reply": reply
        }
