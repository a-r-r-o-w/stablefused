import tempfile

from abc import ABC, abstractmethod
from typing import List


class StoryBookSpeakerBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, messages: List[str], *, yield_files: bool = True) -> List[str]:
        pass


class gTTSStoryBookSpeaker(StoryBookSpeakerBase):
    def __init__(self) -> None:
        super().__init__()
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gTTS is not installed. Please install it using `pip install gTTS`."
            )
        self.gTTS = gTTS

    def __call__(self, messages: List[str], *, yield_files: bool = True) -> List[str]:
        for message in messages:
            tts = self.gTTS(message)

            if yield_files:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    tts.write_to_fp(f)
                    yield f.name
            else:
                files = []
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tts.write_to_fp(f)
                    files.append(f.name)
                return files
