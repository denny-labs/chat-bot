from pathlib import Path
from src.nlu_trainer import NLUTrainer
import asyncio


class NLUInitializer:
    def __init__(self, model_dir: str = "models", intents_path: str = "data/intents.json"):
        self.model_dir = Path(model_dir)
        self.intents_path = intents_path
        self.trainer = NLUTrainer(intents_path=intents_path, model_dir=model_dir)

    async def models_exist(self) -> bool:
        required = [
            self.model_dir / "intent_model.pkl",
            self.model_dir / "vectorizer.pkl",
            self.model_dir / "spacy_nlp"
        ]
        for p in required:
            if not p.exists():
                return False
        return True

    async def ensure_models(self) -> None:
        if not await self.models_exist():
            print("NLU model artifacts missing. Training now — this may take a moment...")
            await self.trainer.train()
            print("NLU training finished and models saved to:", str(self.model_dir))
        else:
            # Models present
            return
