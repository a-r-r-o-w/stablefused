import numpy as np
import torch

from dataclasses import dataclass
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Union

from stablefused.diffusion import BaseDiffusion
from stablefused.typing import PromptType, OutputType, SchedulerType, UNetType
from stablefused.utils import lerp, slerp


@dataclass
class LatentWalkConfig:
    """
    Configuration class for running inference using LatentWalkDiffusion.

    Parameters
    ----------
    prompt: PromptType
        Text prompt to condition on.
    latent: torch.FloatTensor
        Latent to start from.
    strength: float
        The strength of the latent modification, controlling the amount of noise added.
    num_inference_steps: int
        Number of diffusion steps to run.
    guidance_scale: float
        Guidance scale encourages the model to generate images following the prompt
        closely, albeit at the cost of image quality.
    guidance_rescale: float
        Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    negative_prompt: Optional[PromptType]
        Negative text prompt to uncondition on.
    output_type: str
        Type of output to return. One of ["latent", "pil", "pt", "np"].
    return_latent_history: bool
        Whether to return the latent history. If True, return list of all latents
        generated during diffusion steps.
    """

    prompt: PromptType
    latent: torch.FloatTensor
    strength: float = 0.2
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.7
    negative_prompt: Optional[PromptType] = None
    output_type: str = "pil"
    return_latent_history: bool = False


@dataclass
class LatentWalkInterpolateConfig:
    """
    Configuration class for running interpolation using LatentWalkDiffusion.

    Parameters
    ----------
    prompt: List[str]
        List of text prompts to condition on.
    latent: Optional[torch.FloatTensor]
        Latents to interpolate between. If None, latents are generated from noise
        but image_height and image_width must be provided.
    image_height: Optional[int]
        Height of image to generate.
    image_width: Optional[int]
        Width of image to generate.
    num_inference_steps: int
        Number of diffusion steps to run.
    interpolation_steps: Union[int, List[int]]
        Number of interpolation steps to run.
    guidance_scale: float
        Guidance scale encourages the model to generate images following the prompt
        closely, albeit at the cost of image quality.
    guidance_rescale: float
        Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    negative_prompt: Optional[List[str]]
        Negative text prompts to uncondition on.
    output_type: str
        Type of output to return. One of ["latent", "pil", "pt", "np"].
    return_latent_history: bool
        Whether to return the latent history. If True, return list of all latents
        generated during diffusion steps.
    embedding_interpolation_type: str
        Type of interpolation to run for text embeddings. One of ["lerp", "slerp"].
    latent_interpolation_type: str
        Type of interpolation to run for latents. One of ["lerp", "slerp"].
    """

    prompt: List[str] = None
    latent: Optional[torch.FloatTensor] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    num_inference_steps: int = 50
    interpolation_steps: Union[int, List[int]] = 8
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.7
    negative_prompt: Optional[List[str]] = None
    output_type: str = "pil"
    return_latent_history: bool = False
    embedding_interpolation_type: str = "lerp"
    latent_interpolation_type: str = "slerp"


class LatentWalkDiffusion(BaseDiffusion):
    def __init__(
        self,
        model_id: str = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
        vae: AutoencoderKL = None,
        unet: UNetType = None,
        scheduler: SchedulerType = None,
        torch_dtype: torch.dtype = torch.float32,
        device="cuda",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id=model_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            device=device,
            *args,
            **kwargs,
        )

    def modify_latent(
        self,
        latent: torch.FloatTensor,
        strength: float,
    ) -> torch.FloatTensor:
        """
        Modify a latent vector by adding noise.

        Parameters
        ----------
        latent: torch.FloatTensor
            The input latent vector to modify.
        strength: float
            The strength of the modification, controlling the amount of noise added.

        Returns
        -------
        torch.FloatTensor
            Modified latent vector.
        """
        noise = self.random_tensor(latent.shape)
        new_latent = (1 - strength) * latent + strength * noise
        new_latent = (new_latent - new_latent.mean()) / new_latent.std()
        return new_latent

    def embedding_to_latent(
        self,
        embedding: torch.FloatTensor,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_rescale: float,
        latent: torch.FloatTensor,
        return_latent_history: bool = False,
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Generate latent by conditioning on prompt embedding using diffusion.

        Parameters
        ----------
        embedding: torch.FloatTensor
            Embedding of text prompt.
        num_inference_steps: int
            Number of diffusion steps to run.
        guidance_scale: float
            Guidance scale encourages the model to generate images following the prompt
            closely, albeit at the cost of image quality.
        guidance_rescale: float
            Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        latent: torch.FloatTensor
            Latent to start diffusion from.
        return_latent_history: bool
            Whether to return latent history. If True, return list of all latents
            generated during diffusion steps.

        Returns
        -------
        Union[torch.FloatTensor, List[torch.FloatTensor]]
            Latent generated by diffusion. If return_latent_history is True, return list of
            all latents generated during diffusion steps.
        """

        latent = latent.to(self.device)

        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Scale the latent noise by the standard deviation required by the scheduler
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
            noise_prediction = self.classifier_free_guidance(
                noise_prediction, guidance_scale, guidance_rescale
            )

            # Update latent
            latent = self.scheduler.step(
                noise_prediction, timestep, latent, return_dict=False
            )[0]

            if return_latent_history:
                latent_history.append(latent)

        return torch.stack(latent_history) if return_latent_history else latent

    def interpolate_embedding(
        self,
        embedding: torch.FloatTensor,
        interpolation_steps: Union[int, List[int]],
        interpolation_type: str,
    ) -> torch.FloatTensor:
        """
        Interpolate based on interpolation type.

        Parameters
        ----------
        embedding: torch.FloatTensor
            Embedding of text prompt.
        interpolation_steps: Union[int, List[int]]
            Number of interpolation steps to run.
        embedding_interpolation_type: str
            Type of interpolation to run. One of ["lerp", "slerp"].

        Returns
        -------
        torch.FloatTensor
            Interpolated embedding.
        """

        if interpolation_type == "lerp":
            interpolation_fn = lerp
        elif interpolation_type == "slerp":
            interpolation_fn = slerp
        else:
            raise ValueError(
                f"embedding_interpolation_type must be one of ['lerp', 'slerp'], got {interpolation_type}."
            )

        # Split embedding into unconditional and text embeddings
        unconditional_embedding, text_embedding = embedding.chunk(2)
        steps = (
            torch.linspace(0, 1, interpolation_steps, dtype=embedding.dtype)
            .cpu()
            .numpy()
        )
        steps = np.expand_dims(steps, axis=tuple(range(1, text_embedding.ndim)))
        interpolations = []

        # Interpolate between text embeddings
        # TODO: Think of a better way of doing this
        # See if it can be done parallelly instead
        for i in range(text_embedding.shape[0] - 1):
            interpolations.append(
                interpolation_fn(
                    text_embedding[i], text_embedding[i + 1], steps
                ).squeeze(dim=1)
            )
        interpolations = torch.cat(interpolations)

        # TODO: Think of a better way of doing this
        # It can be done because all unconditional embeddings are the same
        single_unconditional_embedding = unconditional_embedding[0].unsqueeze(dim=0)
        unconditional_embedding = single_unconditional_embedding.repeat(
            interpolations.shape[0], 1, 1
        )
        interpolations = torch.cat([unconditional_embedding, interpolations])

        return interpolations

    def interpolate_latent(
        self,
        latent: torch.FloatTensor,
        interpolation_steps: Union[int, List[int]],
        interpolation_type: str,
    ) -> torch.FloatTensor:
        """
        Interpolate latent based on interpolation type.

        Parameters
        ----------
        latent: torch.FloatTensor
            Latent to interpolate.
        interpolation_steps: Union[int, List[int]]
            Number of interpolation steps to run.
        latent_interpolation_type: str
            Type of interpolation to run. One of ["lerp", "slerp"].

        Returns
        -------
        torch.FloatTensor
            Interpolated latent.
        """

        if interpolation_type == "lerp":
            interpolation_fn = lerp
        elif interpolation_type == "slerp":
            interpolation_fn = slerp

        steps = (
            torch.linspace(0, 1, interpolation_steps, dtype=latent.dtype).cpu().numpy()
        )
        steps = np.expand_dims(steps, axis=tuple(range(1, latent.ndim)))
        interpolations = []

        # Interpolate between latents
        # TODO: Think of a better way of doing this
        # See if it can be done parallelly instead
        for i in range(latent.shape[0] - 1):
            interpolations.append(
                interpolation_fn(latent[i], latent[i + 1], steps).squeeze(dim=1)
            )

        return torch.cat(interpolations)

    @torch.no_grad()
    def __call__(
        self,
        config: LatentWalkConfig,
    ) -> OutputType:
        """
        Run inference by conditioning on text prompt starting from provided latent tensor.

        Parameters
        ----------
        config: LatentWalkConfig
            Configuration for running inference using LatentWalkDiffusion.

        Returns
        -------
        OutputType
            Generated output based on output_type.
        """

        prompt = config.prompt
        latent = config.latent
        strength = config.strength
        num_inference_steps = config.num_inference_steps
        guidance_scale = config.guidance_scale
        guidance_rescale = config.guidance_rescale
        negative_prompt = config.negative_prompt
        output_type = config.output_type
        return_latent_history = config.return_latent_history

        # Validate input
        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
        )

        # Generate embedding to condition on prompt and uncondition on negative prompt
        embedding = self.prompt_to_embedding(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Modify latent
        latent = self.modify_latent(latent, strength)

        # Run inference
        latent = self.embedding_to_latent(
            embedding=embedding,
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

    @torch.no_grad()
    def interpolate(
        self,
        config: LatentWalkInterpolateConfig,
    ) -> OutputType:
        """
        Run inference by conditioning on text prompts and interpolating between them.

        Parameters
        ----------
        config: LatentWalkInterpolateConfig
            Configuration for running interpolation using LatentWalkDiffusion.

        Returns
        -------
        OutputType
            Generated output based on output_type.
        """

        prompt = config.prompt
        latent = config.latent
        image_height = config.image_height
        image_width = config.image_width
        num_inference_steps = config.num_inference_steps
        interpolation_steps = config.interpolation_steps
        guidance_scale = config.guidance_scale
        guidance_rescale = config.guidance_rescale
        negative_prompt = config.negative_prompt
        output_type = config.output_type
        return_latent_history = config.return_latent_history
        embedding_interpolation_type = config.embedding_interpolation_type
        latent_interpolation_type = config.latent_interpolation_type

        # Validate input
        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_height=image_height,
            image_width=image_width,
        )

        # There should be atleast 2 prompts to run interpolation
        if not isinstance(prompt, list):
            raise ValueError(f"prompt must be a list of strings, not {type(prompt)}")
        if len(prompt) < 2:
            raise ValueError(
                f"prompt must be a list of at least 2 strings, not {len(prompt)}"
            )
        if isinstance(interpolation_steps, int):
            pass
            # interpolation_steps = [interpolation_steps] * (len(prompt) - 1)
        elif isinstance(interpolation_steps, list):
            if len(interpolation_steps) != len(prompt) - 1:
                raise ValueError(
                    f"interpolation_steps must be a list of length len(prompt) - 1, not {len(interpolation_steps)}"
                )
            raise NotImplementedError(
                "interpolation_steps as a list is not yet implemented"
            )
        else:
            raise ValueError(
                f"interpolation_steps must be an int or list, not {type(interpolation_steps)}"
            )

        if latent is None:
            shape = (
                len(prompt),
                self.unet.config.in_channels,
                image_height // self.vae_scale_factor,
                image_width // self.vae_scale_factor,
            )
            latent = self.random_tensor(shape)
        elif len(prompt) != latent.shape[0]:
            raise ValueError(
                f"prompt and latent must be of the same length, not {len(prompt)} and {latent.shape[0]}"
            )

        # Generate embedding to condition on prompt and uncondition on negative prompt
        embedding = self.prompt_to_embedding(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Interpolate between embeddings
        interpolated_embedding = self.interpolate_embedding(
            embedding=embedding,
            interpolation_steps=interpolation_steps,
            interpolation_type=embedding_interpolation_type,
        )

        # Interpolate between latents
        interpolated_latent = self.interpolate_latent(
            latent=latent,
            interpolation_steps=interpolation_steps,
            interpolation_type=latent_interpolation_type,
        )

        # Run inference
        latent = self.embedding_to_latent(
            embedding=interpolated_embedding,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            latent=interpolated_latent,
            return_latent_history=return_latent_history,
        )

        return self.resolve_output(
            latent=latent,
            output_type=output_type,
            return_latent_history=return_latent_history,
        )
