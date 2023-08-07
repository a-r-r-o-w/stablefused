import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Union

from ..utils import (
    denormalize,
    normalize,
    numpy_to_pil,
    numpy_to_pt,
    pil_to_numpy,
    pt_to_numpy,
)


class BaseDiffusion:
    def __init__(
        self,
        model_id: str = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
        vae: AutoencoderKL = None,
        unet: UNet2DConditionModel = None,
        scheduler: KarrasDiffusionSchedulers = None,
        device="cuda",
    ) -> None:
        self.model_id = model_id
        self.device = device

        if model_id is None:
            if (
                tokenizer is None
                or text_encoder is None
                or vae is None
                or unet is None
                or scheduler is None
            ):
                raise ValueError(
                    "Either (`model_id`) or (`tokenizer`, `text_encoder`, `vae`, `unet` and `scheduler`) must be provided."
                )
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder.to(device)
            self.vae = vae.to(device)
            self.unet = unet.to(device)
            self.scheduler = scheduler
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_id, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_id, subfolder="text_encoder"
            ).to(device)
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(
                device
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet"
            ).to(device)
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @staticmethod
    def validate_input(
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image_height: int = None,
        image_width: int = None,
        start_step: int = None,
        num_inference_steps: int = None,
    ) -> None:
        if image_height is not None and image_width is not None:
            if image_height % 8 != 0 or image_width % 8 != 0:
                raise ValueError(
                    "`image_height` and `image_width` must a multiple of 8"
                )
        if negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "Type of `prompt` and `negative_prompt` must be the same"
                )
            if isinstance(prompt, list) and len(prompt) != len(negative_prompt):
                raise ValueError(
                    "Length of `prompt` list and `negative_prompt` list should match"
                )
        if start_step is not None:
            if num_inference_steps is None:
                raise ValueError(
                    "`num_inference_steps` must be provided if `start_step` is provided"
                )
            if start_step < 0 or start_step >= num_inference_steps:
                raise ValueError(
                    "`start_step` must be in the range [0, `num_inference_steps` - 1]"
                )

    def latent_to_image(
        self, latent: torch.FloatTensor, output_type: str
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError("`output_type` must be one of [`pt`, `np`, `pil`]")

        image = self.vae.decode(
            latent / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = denormalize(image)

        if output_type == "pt":
            return image

        image = pt_to_numpy(image)

        if output_type == "np":
            return image

        image = numpy_to_pil(image)
        return image

    def image_to_latent(
        self,
        image: Union[Image.Image, List[Image.Image]],
    ) -> torch.FloatTensor:
        if (
            not isinstance(image, Image.Image)
            and not isinstance(image, list)
            and not isinstance(image, np.ndarray)
            and not isinstance(image, torch.Tensor)
        ):
            raise TypeError(
                "`image` type must be one of (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`). Other types are not supported yet"
            )
        if isinstance(image, Image.Image):
            image: List[Image.Image] = [image]

        if isinstance(image[0], Image.Image):
            image: np.ndarray = pil_to_numpy(image)

        if isinstance(image[0], np.ndarray):
            image: torch.FloatTensor = numpy_to_pt(image)

        image = image.to(self.device)
        latent = (
            self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        )

        return latent
