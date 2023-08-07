import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Union

from .base_diffusion import BaseDiffusion


class TextToImageDiffusion(BaseDiffusion):
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
        super().__init__(
            model_id, tokenizer, text_encoder, vae, unet, scheduler, device
        )

    def embedding_to_latent(
        self,
        embedding: torch.FloatTensor,
        image_height: int,
        image_width: int,
        num_inference_steps: int,
        guidance_scale: float,
        latent: Optional[torch.FloatTensor] = None,
        return_latent_history: bool = False,
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """Convert CLIP embedding(s) to latent vector(s)."""

        use_classifier_free_guidance = guidance_scale > 1.0

        if latent is None:
            shape = (
                embedding.shape[0] // 2,
                self.unet.config.in_channels,
                image_height // self.vae_scale_factor,
                image_width // self.vae_scale_factor,
            )
            latent = torch.randn(shape)
        latent = latent.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        latent = latent * self.scheduler.init_noise_sigma
        latent_history = [latent]

        for i, timestep in tqdm(list(enumerate(timesteps))):
            latent_model_input = (
                torch.cat([latent] * 2) if use_classifier_free_guidance else latent
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            noise_prediction = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=embedding,
                return_dict=False,
            )[0]

            if use_classifier_free_guidance:
                noise_unconditional, noise_prompt = noise_prediction.chunk(2)
                noise_prediction = noise_unconditional + guidance_scale * (
                    noise_prompt - noise_unconditional
                )

            latent = self.scheduler.step(
                noise_prediction, timestep, latent, return_dict=False
            )[0]

            if return_latent_history:
                latent_history.append(latent)

        return latent_history if return_latent_history else latent

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image_height: int = 512,
        image_width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt=None,
        latent: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_latent_history: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, List[Image.Image]]:
        """Generate image(s) from prompt(s)."""

        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_height=image_height,
            image_width=image_width,
        )
        embedding = self.prompt_to_embedding(prompt, guidance_scale, negative_prompt)
        latent = self.embedding_to_latent(
            embedding=embedding,
            image_height=image_height,
            image_width=image_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latent=latent,
            return_latent_history=return_latent_history,
        )

        if output_type == "latent":
            return latent

        if return_latent_history:
            image: np.ndarray = np.array(
                [self.latent_to_image(l, output_type) for l in tqdm(latent)]
            )
            dims = len(image.shape)
            image = np.transpose(image, (1, 0, *range(2, dims)))
        else:
            image = self.latent_to_image(latent, output_type)
        return image

    generate = __call__
