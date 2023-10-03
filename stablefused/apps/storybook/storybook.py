import json
import os

from stablefused import TextToImageDiffusion
from stablefused.apps.storybook import StoryBookAuthorBase
from typing import Dict, List, Union


class StoryBook:
    _artist_call_attributes = [
        "image_height",
        "image_width",
        "num_inference_steps",
        "guidance_scale",
        "guidance_rescale",
        "negative_prompt",
    ]

    def __init__(
        self,
        author: StoryBookAuthorBase,
        artist: TextToImageDiffusion,
        speaker=None,
        *,
        config: Union[str, Dict[str, str]] = "config/default_1_shot.json",
    ) -> None:
        self.author = author
        self.artist = artist
        self.config = config

        self._process_config(self.config)

    def _process_config(self, config: Union[str, Dict[str, str]]) -> None:
        if isinstance(config, str):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, config)

            with open(config_path, "r") as f:
                config = json.load(f)

        self.artist_call_kwargs = {
            k: v for k, v in config.items() if k in self._artist_call_attributes
        }
        self.negative_prompt = self.artist_call_kwargs.pop("negative_prompt", None)
        self.messages: List[Dict[str, str]] = config.get("messages")

    def create_prompt(self, role: str, content: str) -> Dict[str, str]:
        return {"role": role, "content": content}

    def validate_output(self, output: str) -> List[Dict[str, str]]:
        try:
            output_list = json.loads(output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Output is not a valid JSON: {str(e)}")

        if not isinstance(output_list, list):
            raise ValueError("Output must be a list of dictionaries.")

        for item in output_list:
            if not isinstance(item, dict):
                raise ValueError("Each item in the list must be a dictionary.")

            if "story" not in item or "prompt" not in item:
                raise ValueError(
                    "Each dictionary must contain 'story' and 'prompt' keys."
                )

            if not isinstance(item["story"], str) or not isinstance(
                item["prompt"], str
            ):
                raise ValueError("'story' and 'prompt' values must be strings.")

        return output_list

    def __call__(
        self,
        user_prompt: str,
        *,
        display_captions: bool = True,
        output_filename="output.mp4",
    ) -> None:
        messages = [*self.messages, self.create_prompt("user", user_prompt)]
        storybook = self.author(messages)
        storybook = self.validate_output(storybook)
        prompt = [item["prompt"] for item in storybook]
        negative_prompt = (
            [self.negative_prompt] * len(prompt)
            if self.negative_prompt is not None
            else None
        )
        images = self.artist(
            prompt=prompt, negative_prompt=negative_prompt, **self.artist_call_kwargs
        )
        return storybook, images
