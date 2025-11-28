# import json
# from pathlib import Path
# from typing import List, Tuple
# import joblib
# import asyncio

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# import spacy


# class NLUTrainer:
#     def __init__(self, intents_path: str = "data/intents.json", model_dir: str = "models"):
#         self.intents_path = Path(intents_path)
#         self.model_dir = Path(model_dir)
#         self.model_dir.mkdir(parents=True, exist_ok=True)

#     async def load_intents(self) -> List[dict]:
#         with open(self.intents_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         return data.get("intents", [])

#     async def train_intent_model(self) -> Tuple:
#         intents = await self.load_intents()
#         X: List[str] = []
#         y: List[str] = []

#         for intent_obj in intents:
#             intent = intent_obj["intent"]
#             for p in intent_obj.get("patterns", []):
#                 X.append(p)
#                 y.append(intent)

#         if not X:
#             raise RuntimeError("No training examples found in data/intents.json")

#         vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
#         X_vec = vectorizer.fit_transform(X)

#         model = LogisticRegression(max_iter=1000)
#         model.fit(X_vec, y)

#         # Save artifacts
#         joblib.dump(model, self.model_dir / "intent_model.pkl")
#         joblib.dump(vectorizer, self.model_dir / "vectorizer.pkl")

#         return model, vectorizer

#     async def train_spacy_pipeline(self):
#         # Create a minimal spaCy pipeline and save it.
#         # You can replace this by a trained NER if you want; for now we create a lightweight blank pipeline.
#         nlp = spacy.blank("en")

#         # Optionally add components or matchers here.
#         # Keep it simple: users can install en_core_web_sm and replace blank with real model.
#         nlp.to_disk(self.model_dir / "spacy_nlp")

#     async def train(self):
#         # Run training steps
#         await self.train_intent_model()
#         await self.train_spacy_pipeline()
#         return True

import json
import os
import random
import spacy
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib


class NLUTrainer:
    def __init__(self):
        self.intents_path = os.path.join("data", "intents.json")
        self.models_dir = "models"

        # Files to save
        self.vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
        self.intent_model_path = os.path.join(self.models_dir, "intent_model.pkl")
        self.spacy_model_path = os.path.join(self.models_dir, "spacy_nlp")

        # Make sure model directory exists
        os.makedirs(self.models_dir, exist_ok=True)

    def load_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["intents"]

    def prepare_intent_dataset(self, intents):
        X = []
        y = []

        for item in intents:
            intent = item["intent"]
            patterns = item["patterns"]

            for p in patterns:
                X.append(p.lower().strip())
                y.append(intent)

        return X, y

    def train_intent_classifier(self, X, y):
        print("\n🔹 Training TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        print("🔹 Training LinearSVC Intent Model...")
        model = LinearSVC()
        model.fit(X_vec, y)

        print("✅ Intent classifier trained successfully!")
        return vectorizer, model

    def build_spacy_ner(self):
        """
        Creates a spaCy NLP model using EntityRuler.
        You can update patterns.json later for custom entities.
        """
        print("\n🔹 Creating spaCy Entity Pipeline...")

        nlp = spacy.blank("en")  # empty model
        ruler = nlp.add_pipe("entity_ruler")

        # Basic academic entities (can expand later)
        patterns = [
            {"label": "SUBJECT", "pattern": "machine learning"},
            {"label": "SUBJECT", "pattern": "python"},
            {"label": "SUBJECT", "pattern": "calculus"},
            {"label": "SUBJECT", "pattern": "dbms"},
            {"label": "SUBJECT", "pattern": "photosynthesis"},
            {"label": "SUBJECT", "pattern": "thermodynamics"},
        ]

        ruler.add_patterns(patterns)

        print("🔹 Saving spaCy model...")
        nlp.to_disk(self.spacy_model_path)

        print("✅ spaCy NER model saved!")
        return nlp

    def save_models(self, vectorizer, intent_model):
        print("\n🔹 Saving sklearn models...")
        joblib.dump(vectorizer, self.vectorizer_path)
        joblib.dump(intent_model, self.intent_model_path)
        print("✅ Intent model + vectorizer saved!")

    def run(self):
        print("🚀 Starting NLU Training Process...")

        # Step 1 — Load intents.json
        intents = self.load_intents()

        # Step 2 — Prepare dataset
        X, y = self.prepare_intent_dataset(intents)
        print(f"🔹 Loaded {len(X)} training utterances across {len(set(y))} intents.")

        # Step 3 — Train intent classifier
        vectorizer, intent_model = self.train_intent_classifier(X, y)

        # Step 4 — Save intent model + vectorizer
        self.save_models(vectorizer, intent_model)

        # Step 5 — Create spaCy NER pipeline
        self.build_spacy_ner()

        print("\n🎉 TRAINING COMPLETE!")
        print("Models saved inside: /models/")
        print("----------------------------------")


if __name__ == "__main__":
    trainer = NLUTrainer()
    trainer.run()
