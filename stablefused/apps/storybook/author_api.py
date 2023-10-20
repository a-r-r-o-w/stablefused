from abc import ABC, abstractmethod
from typing import Dict, List

from stablefused import LazyImporter


class StoryBookAuthorBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        pass


class G4FStoryBookAuthor(StoryBookAuthorBase):
    def __init__(self, model_id: str = "gpt-3.5-turbo") -> None:
        super().__init__()
        self.model_id = model_id
        self.g4f = LazyImporter("g4f")

    def __call__(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        g4f = self.g4f.import_module(
            import_error_message="g4f is not installed. Please install it using `pip install g4f`."
        )
        return g4f.ChatCompletion.create(model=self.model_id, messages=messages)
