from pathlib import Path
from typing import List, Tuple, Optional
import joblib
import json
import spacy


class NLUPipeline:
    def __init__(self, model_dir: str = "models", intents_path: str = "data/intents.json"):
        self.model_dir = Path(model_dir)
        self.intents_path = Path(intents_path)

        # Load saved artifacts
        self.model = joblib.load(self.model_dir / "intent_model.pkl")
        self.vectorizer = joblib.load(self.model_dir / "vectorizer.pkl")
        try:
            # Prefer a saved pipeline
            self.nlp = spacy.load(self.model_dir / "spacy_nlp")
        except Exception:
            # Fallback to blank
            self.nlp = spacy.blank("en")

        # Load intents for mapping to responses
        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)

    async def predict_intent(self, text: str) -> Tuple[str, float]:
        X = self.vectorizer.transform([text])
        # intent label
        intent = self.model.predict(X)[0]
        # confidence if available
        confidence = 1.0
        try:
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba.max())
        except Exception:
            # model may not support predict_proba
            confidence = 1.0
        return intent, confidence

    async def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    async def get_reply(self, intent: str) -> str:
        for intent_obj in self.intents_data.get("intents", []):
            if intent_obj.get("intent") == intent:
                responses = intent_obj.get("responses") or intent_obj.get("responses", [])
                if responses:
                    return responses[0]
        # Fallback
        return "Sorry, I didn't understand that."
