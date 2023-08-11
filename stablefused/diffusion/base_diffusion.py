import numpy as np
import torch

from PIL import Image
from abc import ABC, abstractmethod
from diffusers import AutoencoderKL, SchedulerMixin, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, List, Optional, Union

from stablefused.utils import (
    cache_model,
    denormalize,
    load_model,
    normalize,
    numpy_to_pil,
    numpy_to_pt,
    pil_to_numpy,
    pt_to_numpy,
)


class BaseDiffusion(ABC):
    def __init__(
        self,
        model_id: str = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
        vae: AutoencoderKL = None,
        unet: UNet2DConditionModel = None,
        scheduler: KarrasDiffusionSchedulers = None,
        name: str = None,
        torch_dtype: torch.dtype = torch.float32,
        device="cuda",
    ) -> None:
        self.device = device

        if model_id is None:
            if (
                tokenizer is None
                or text_encoder is None
                or vae is None
                or unet is None
                or scheduler is None
                or name is None
            ):
                raise ValueError(
                    "Either (`model_id`) or (`tokenizer`, `text_encoder`, `vae`, `unet`, `scheduler`, `name`) must be provided."
                )

            model = cache_model(
                name=name,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                scheduler=scheduler,
            )
        else:
            model = load_model(model_id, torch_dtype=torch_dtype)

        self.model_id = model.model_id
        self.tokenizer: CLIPTokenizer = model.tokenizer
        self.text_encoder: CLIPTextModel = model.text_encoder
        self.vae: AutoencoderKL = model.vae
        self.unet: UNet2DConditionModel = model.unet
        self.scheduler: SchedulerMixin = model.scheduler

        self.to(self.device)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def to(self, device: str) -> None:
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)

    def enable_attention_slicing(self, slice_size: Optional[int] = -1) -> None:
        """
        Enable attention slicing. By default, the attention head is sliced in half.
        This is a good tradeoff between memory and performance.

        Parameters
        ----------
        slice_size: int
            The size of the attention slice. If -1, the attention head is sliced in
            half. If None, attention slicing is disabled.
        """
        if slice_size == -1:
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self) -> None:
        """Disable attention slicing."""
        self.unet.set_attention_slice(None)

    def enable_slicing(self) -> None:
        """
        Allow tensor slicing for vae decode step. This will cause the vae to split
        the input tensor to compute decoding in multiple steps. This will save
        memory and allow for larger batch sizes, but will affect performance slightly.
        """
        self.vae.enable_slicing()

    def disable_slicing(self) -> None:
        """Disable tensor slicing for vae decode step."""
        self.vae.disable_slicing()

    def enable_tiling(self) -> None:
        """
        Allow tensor tiling for vae. This will cause the vae to split the input tensor
        into tiles to compute encoding/decoding in several steps. This will save a large
        amount of memory and allow processing larger images, but will affect performance.
        """
        self.vae.enable_tiling()

    def disable_tiling(self) -> None:
        """Disable tensor tiling for vae."""
        self.vae.disable_tiling()

    @staticmethod
    def validate_input(
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image_height: int = None,
        image_width: int = None,
        start_step: int = None,
        num_inference_steps: int = None,
        strength: float = None,
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
        if strength is not None:
            if strength < 0 or strength > 1:
                raise ValueError("`strength` must be in the range [0.0, 1.0]")

    @abstractmethod
    def embedding_to_latent(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def prompt_to_embedding(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.FloatTensor:
        """Convert prompt(s) to a CLIP embedding(s)."""

        if negative_prompt is not None:
            assert type(prompt) is type(negative_prompt)

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if negative_prompt is not None:
                negative_prompt = [negative_prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise TypeError("`prompt` must be a string or a list of strings")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        text_embedding = self.text_encoder(
            text_input.input_ids.to(self.device), attention_mask=attention_mask
        )[0]

        if negative_prompt is None:
            unconditioning_input = [""] * batch_size
        else:
            unconditioning_input = negative_prompt

        unconditioning_input = self.tokenizer(
            unconditioning_input,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        unconditional_embedding = self.text_encoder(
            unconditioning_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )[0]
        embedding = torch.cat([unconditional_embedding, text_embedding])

        return embedding

    def do_classifier_free_guidance(
        self,
        noise_prediction: torch.FloatTensor,
        guidance_scale: float,
        guidance_rescale: float,
    ) -> torch.FloatTensor:
        """Apply classifier-free guidance to noise prediction."""

        # Perform guidance
        noise_unconditional, noise_prompt = noise_prediction.chunk(2)
        noise_prediction = noise_unconditional + guidance_scale * (
            noise_prompt - noise_unconditional
        )

        # Rescale noise prediction according to guidance scale
        # Based on findings in Section 3.4  of [Common Diffusion Noise Schedules and Sample
        # Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        std_prompt = noise_prompt.std(
            dim=list(range(1, noise_prompt.ndim)), keepdim=True
        )
        std_prediction = noise_prediction.std(
            dim=list(range(1, noise_prediction.ndim)), keepdim=True
        )
        noise_prediction_rescaled = noise_prediction * (std_prompt / std_prediction)
        noise_prediction = (
            noise_prediction * (1 - guidance_rescale)
            + noise_prediction_rescaled * guidance_rescale
        )

        return noise_prediction

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

    def resolve_output(
        self,
        latent: torch.FloatTensor,
        output_type: str,
        return_latent_history: bool,
    ) -> Union[torch.Tensor, np.ndarray, Image.Image, List[Image.Image]]:
        if output_type == "latent":
            return latent

        if return_latent_history:
            latent = torch.transpose(latent, 0, 1)
            image = [self.latent_to_image(l, output_type) for l in latent]
            if output_type == "pt":
                image = torch.stack(image)
            elif output_type == "np":
                image = np.stack(image)
        else:
            image = self.latent_to_image(latent, output_type)
        return image
