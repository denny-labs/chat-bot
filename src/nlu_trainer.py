import json
from pathlib import Path
from typing import List, Tuple
import joblib
import asyncio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import spacy


class NLUTrainer:
    def __init__(self, intents_path: str = "data/intents.json", model_dir: str = "models"):
        self.intents_path = Path(intents_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    async def load_intents(self) -> List[dict]:
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", [])

    async def train_intent_model(self) -> Tuple:
        intents = await self.load_intents()
        X: List[str] = []
        y: List[str] = []

        for intent_obj in intents:
            intent = intent_obj["intent"]
            for p in intent_obj.get("patterns", []):
                X.append(p)
                y.append(intent)

        if not X:
            raise RuntimeError("No training examples found in data/intents.json")

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, y)

        # Save artifacts
        joblib.dump(model, self.model_dir / "intent_model.pkl")
        joblib.dump(vectorizer, self.model_dir / "vectorizer.pkl")

        return model, vectorizer

    async def train_spacy_pipeline(self):
        # Create a minimal spaCy pipeline and save it.
        # You can replace this by a trained NER if you want; for now we create a lightweight blank pipeline.
        nlp = spacy.blank("en")

        # Optionally add components or matchers here.
        # Keep it simple: users can install en_core_web_sm and replace blank with real model.
        nlp.to_disk(self.model_dir / "spacy_nlp")

    async def train(self):
        # Run training steps
        await self.train_intent_model()
        await self.train_spacy_pipeline()
        return True
