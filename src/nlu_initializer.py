# from pathlib import Path
# from src.nlu_trainer import NLUTrainer
# import asyncio


# class NLUInitializer:
#     def __init__(self, model_dir: str = "models", intents_path: str = "data/intents.json"):
#         self.model_dir = Path(model_dir)
#         self.intents_path = intents_path
#         self.trainer = NLUTrainer(intents_path=intents_path, model_dir=model_dir)

#     async def models_exist(self) -> bool:
#         required = [
#             self.model_dir / "intent_model.pkl",
#             self.model_dir / "vectorizer.pkl",
#             self.model_dir / "spacy_nlp"
#         ]
#         for p in required:
#             if not p.exists():
#                 return False
#         return True

#     async def ensure_models(self) -> None:
#         if not await self.models_exist():
#             print("NLU model artifacts missing. Training now — this may take a moment...")
#             await self.trainer.train()
#             print("NLU training finished and models saved to:", str(self.model_dir))
#         else:
#             # Models present
#             return

import joblib
import spacy
from pathlib import Path


class NLUInitializer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)

        self.intent_model_path = self.model_dir / "intent_model.pkl"
        self.vectorizer_path = self.model_dir / "vectorizer.pkl"
        self.spacy_model_path = self.model_dir / "spacy_nlp"

        # Model holders
        self.vectorizer = None
        self.intent_model = None
        self.spacy_nlp = None

    def load_intent_model(self):
        if not self.intent_model_path.exists():
            raise FileNotFoundError("intent_model.pkl not found. Train the model first.")

        self.intent_model = joblib.load(self.intent_model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        print("🔹 Loaded intent classifier + TF-IDF vectorizer.")

    def load_spacy_model(self):
        if not self.spacy_model_path.exists():
            raise FileNotFoundError("spaCy NLP model not found. Train the pipeline first.")

        self.spacy_nlp = spacy.load(self.spacy_model_path)
        print("🔹 Loaded spaCy NER model.")

    def load_all(self):
        """Main loader for all NLU models."""
        self.load_intent_model()
        self.load_spacy_model()

        print("✅ All NLU models loaded successfully.")
        return {
            "intent_model": self.intent_model,
            "vectorizer": self.vectorizer,
            "spacy_nlp": self.spacy_nlp
        }
