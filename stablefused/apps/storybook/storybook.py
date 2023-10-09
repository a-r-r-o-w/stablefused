import json
import numpy as np
import os

from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from typing import Dict, List, Optional, Union

from stablefused import (
    TextToImageConfig,
    TextToImageDiffusion,
    LatentWalkInterpolateConfig,
    LatentWalkDiffusion,
)
from stablefused.apps.storybook import StoryBookAuthorBase, StoryBookSpeakerBase
from stablefused.utils import write_text_on_image


@dataclass
class StoryBookConfig(DataClassJsonMixin):
    """
    Configuration class for running inference with StoryBook.
    """

    prompt: str
    artist_config: Union[TextToImageConfig, LatentWalkInterpolateConfig]
    artist_attributes: str = ""
    messages: List[Dict[str, str]] = None
    display_captions: bool = True
    caption_fontsize: int = 30
    caption_fontfile: str = "arial.ttf"
    caption_padding: int = 10
    frame_duration: int = 1
    num_retries: int = 3
    output_filename: str = "output.mp4"


class StoryBook:
    def __init__(
        self,
        author: StoryBookAuthorBase,
        artist: Union[TextToImageDiffusion, LatentWalkDiffusion],
        speaker: Optional[StoryBookSpeakerBase] = None,
    ) -> None:
        self.author = author
        self.artist = artist
        self.speaker = speaker

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
        self.attributes = config.get("attributes", "")

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
        config: StoryBookConfig,
    ) -> None:
        print(config.to_json(indent=2))
        prompt = config.prompt
        artist_config = config.artist_config
        artist_attributes = config.artist_attributes
        messages = config.messages
        display_captions = config.display_captions
        caption_fontsize = config.caption_fontsize
        caption_fontfile = config.caption_fontfile
        caption_padding = config.caption_padding
        frame_duration = config.frame_duration
        num_retries = config.num_retries
        output_filename = config.output_filename

        if (
            isinstance(artist_config, TextToImageConfig)
            and isinstance(self.artist, LatentWalkDiffusion)
        ) or (
            isinstance(artist_config, LatentWalkInterpolateConfig)
            and isinstance(self.artist, TextToImageDiffusion)
        ):
            raise ValueError(
                "Artist is not compatible with the provided artist config."
            )

        messages.append(self.create_prompt("user", prompt))

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

        prompt = [f"{item['prompt']}, {artist_attributes}" for item in storybook]
        artist_config.prompt = prompt
        artist_config.negative_prompt = (
            [artist_config.negative_prompt] * len(storybook)
            if artist_config.negative_prompt is not None
            else None
        )

        if isinstance(self.artist, TextToImageDiffusion):
            images = self.artist(artist_config)
        else:
            images = self.artist.interpolate(artist_config)

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
            frame_duration = audioclip.duration / len(prompt)
        else:
            audioclip = None

        if isinstance(self.artist, TextToImageDiffusion):
            video = [
                ImageClip(np.array(image), duration=frame_duration) for image in images
            ]
        else:
            num_frames_per_prompt = config.artist_config.interpolation_steps
            video = []
            for i in range(0, len(images), num_frames_per_prompt):
                current_images = images[i : i + num_frames_per_prompt]
                current_clips = [
                    ImageClip(np.array(image), duration=frame_duration)
                    for image in current_images
                ]
                video.append(concatenate_videoclips(current_clips))

        video: CompositeVideoClip = concatenate_videoclips(video)
        video = video.set_audio(audioclip)
        video = video.set_fps(60)
        video.write_videofile(output_filename)

        return storybook
