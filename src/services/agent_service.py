import asyncio
from typing import Tuple, List
from src.nlu_initializer import NLUInitializer
from src.nlu_pipeline import NLUPipeline


class AgentService:
    """
    Manages model initialization and holds the NLUPipeline instance.
    On creation it ensures models exist and loads the pipeline.
    """
    def __init__(self):
        self._pipeline: NLUPipeline | None = None
        self._initializer = NLUInitializer()
        # Start loading models asynchronously
        loop = asyncio.get_event_loop()
        # create_task safe in main thread; if not available, run synchronously fallback
        try:
            loop.create_task(self._ensure_and_load())
        except RuntimeError:
            # If no running loop, run synchronously
            loop.run_until_complete(self._ensure_and_load())

    async def _ensure_and_load(self):
        await self._initializer.ensure_models()
        self._pipeline = NLUPipeline()

    async def wait_until_ready(self):
        # simple wait until pipeline is loaded
        while self._pipeline is None:
            await asyncio.sleep(0.05)
        return True

    async def predict_intent(self, text: str) -> Tuple[str, float]:
        await self.wait_until_ready()
        return await self._pipeline.predict_intent(text)

    async def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        await self.wait_until_ready()
        return await self._pipeline.extract_entities(text)

    async def get_reply(self, intent: str) -> str:
        await self.wait_until_ready()
        return await self._pipeline.get_reply(intent)
