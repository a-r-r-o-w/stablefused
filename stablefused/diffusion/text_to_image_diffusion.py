import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Union

from stablefused.diffusion import BaseDiffusion


class TextToImageDiffusion(BaseDiffusion):
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
        super().__init__(
            model_id=model_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            name=name,
            torch_dtype=torch_dtype,
            device=device,
        )

    def embedding_to_latent(
        self,
        embedding: torch.FloatTensor,
        image_height: int,
        image_width: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_rescale: float,
        latent: Optional[torch.FloatTensor] = None,
        return_latent_history: bool = False,
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Generate latent by conditioning on prompt embedding using diffusion.

        Parameters
        ----------
        embedding: torch.FloatTensor
            CLIP embedding of text prompt.
        image_height: int
            Height of image to generate.
        image_width: int
            Width of image to generate.
        num_inference_steps: int
            Number of diffusion steps to run.
        guidance_scale: float
            Guidance scale encourages the model to generate images following the prompt
            closely, albeit at the cost of image quality.
        guidance_rescale: float
            Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        latent: Optional[torch.FloatTensor]
            Latent to start from. If None, generate latent from noise.
        return_latent_history: bool
            Whether to return latent history. If True, return list of all latents
            generated during diffusion steps.

        Returns
        -------
        Union[torch.FloatTensor, List[torch.FloatTensor]]
            Latent generated by diffusion. If return_latent_history is True, return
            list of all latents generated during diffusion steps.
        """

        # Generate latent from noise if not provided
        if latent is None:
            shape = (
                embedding.shape[0] // 2,
                self.unet.config.in_channels,
                image_height // self.vae_scale_factor,
                image_width // self.vae_scale_factor,
            )
            latent = torch.randn(shape)
        latent = latent.to(self.device)

        # Set number of inference steps in scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Scales the latent noise by the standard deviation required by the scheduler
        latent = latent * self.scheduler.init_noise_sigma
        latent_history = [latent]

        # Diffusion inference loop
        for i, timestep in tqdm(list(enumerate(timesteps))):
            # Duplicate latent to avoid two forward passes to perform classifier free guidance
            latent_model_input = torch.cat([latent] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            # Predict noise
            noise_prediction = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=embedding,
                return_dict=False,
            )[0]

            # Perform classifier free guidance
            noise_prediction = self.do_classifier_free_guidance(
                noise_prediction, guidance_scale, guidance_rescale
            )

            # Update latent
            latent = self.scheduler.step(
                noise_prediction, timestep, latent, return_dict=False
            )[0]

            if return_latent_history:
                latent_history.append(latent)

        return torch.stack(latent_history) if return_latent_history else latent

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image_height: int = 512,
        image_width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.7,
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
        embedding = self.prompt_to_embedding(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        latent = self.embedding_to_latent(
            embedding=embedding,
            image_height=image_height,
            image_width=image_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            latent=latent,
            return_latent_history=return_latent_history,
        )

        return self.resolve_output(
            latent=latent,
            output_type=output_type,
            return_latent_history=return_latent_history,
        )

    generate = __call__
