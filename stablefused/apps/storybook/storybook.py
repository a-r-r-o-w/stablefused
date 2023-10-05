import json
import numpy as np
import os

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)
from stablefused import TextToImageDiffusion
from stablefused.apps.storybook import StoryBookAuthorBase, StoryBookSpeakerBase
from stablefused.utils import write_text_on_image
from typing import Dict, List, Optional, Union


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
        speaker: Optional[StoryBookSpeakerBase] = None,
        *,
        config: Union[str, Dict[str, str]] = "config/default_1_shot.json",
    ) -> None:
        self.author = author
        self.artist = artist
        self.speaker = speaker
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
        frame_duration: int = 1,
        display_captions: bool = True,
        caption_fontsize: int = 30,
        caption_fontfile: str = "arial.ttf",
        caption_padding: int = 10,
        speedup_factor: int = 1,
        num_retries: int = 3,
        output_filename="output.mp4",
    ) -> None:
        messages = [*self.messages, self.create_prompt("user", user_prompt)]

        for i in range(num_retries):
            try:
                storybook = self.author(messages)
                storybook = self.validate_output(storybook)
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Retrying ({i + 1}/{num_retries})...")
                continue
        else:
            raise Exception("Failed to generate storybook. Please try again.")

        prompt = [item["prompt"] for item in storybook]
        negative_prompt = (
            [self.negative_prompt] * len(prompt)
            if self.negative_prompt is not None
            else None
        )

        images = self.artist(
            prompt=prompt, negative_prompt=negative_prompt, **self.artist_call_kwargs
        )

        if display_captions:
            images = [
                write_text_on_image(
                    image,
                    storypart.get("story"),
                    fontfile=caption_fontfile,
                    fontsize=caption_fontsize,
                    padding=caption_padding,
                )
                for image, storypart in zip(images, storybook)
            ]

        if self.speaker is not None:
            stories = [item.get("story") for item in storybook]
            audioclips = []

            for audiofile in self.speaker(stories, yield_files=True):
                audioclips.append(AudioFileClip(audiofile))

            audioclip: CompositeAudioClip = concatenate_audioclips(audioclips)
            frame_duration = audioclip.duration / len(images)
        else:
            audioclip = None

        video = [
            ImageClip(np.array(image), duration=frame_duration) for image in images
        ]
        video = concatenate_videoclips(video)
        video: CompositeVideoClip = video.set_audio(audioclip)
        video: CompositeVideoClip = video.set_fps(60)
        video: CompositeVideoClip = video.fx(vfx.speedx, speedup_factor)
        video.write_videofile(output_filename)

        return storybook
