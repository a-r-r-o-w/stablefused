import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from dataclasses import dataclass
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, List, Optional, Tuple, Union

from stablefused.diffusion import BaseDiffusion
from stablefused.typing import (
    InpaintWalkType,
    ImageType,
    OutputType,
    PromptType,
    SchedulerType,
    UNetType,
)


@dataclass
class InpaintConfig:
    """
    Configuration class for running inference with InpaintDiffusion.

    Parameters
    ----------
    prompt: PromptType
            Text prompt to condition on.
    image: ImageType
        Input image to condition on.
    mask: ImageType
        Input mask to condition on. Must have same height and width as image.
        Must have 1 channel. Must have values between 0 and 1. Values below 0.5
        are treated as 0 and values above 0.5 are treated as 1. Values that are
        1 (white) will be inpainted and values that are 0 (black) will be
        preserved.
    image_height: int
        Height of image to generate. If height of provided image and image_height
        do not match, image will be resized to image_height using PIL Lanczos method.
    image_width: int
        Width of image to generate. If width of provided image and image_width
        do not match, image will be resized to image_width using PIL Lanczos method.
    num_inference_steps: int
        Number of diffusion steps to run.
    start_step: int
        Step to start diffusion from. The higher the value, the more similar the generated
        image will be to the input image.
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
    image: ImageType
    mask: ImageType
    image_height: int = 512
    image_width: int = 512
    num_inference_steps: int = 50
    start_step: int = 0
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.7
    negative_prompt: Optional[PromptType] = None
    output_type: str = "pil"
    return_latent_history: bool = False


@dataclass
class InpaintWalkConfig:
    """
    Configuration class for running inpaint walking with InpaintDiffusion.

    Parameters
    ----------
    prompt: PromptType
        Text prompt to condition on.
    image: Image.Image
        Input image to condition on for inpainting.
    walk_type: Union[InpaintWalkType, List[InpaintWalkType]]
        Type of walk to perform. If List[InpaintWalkType], must have length of
        num_inpainting_steps.
    image_height: int
        Height of image to generate. If height of provided image and image_height
        do not match, image will be resized to image_height using PIL Lanczos method.
    image_width: int
        Width of image to generate. If width of provided image and image_width
        do not match, image will be resized to image_width using PIL Lanczos method.
    height_translation_per_step: int
        Number of pixels to translate image up/down per step.
    width_translation_per_step: int
        Number of pixels to translate image left/right per step.
    translation_factor: Optional[float]
        Factor to translate image by. If provided, overrides height_translation_per_step
        and width_translation_per_step. Must be between 0 and 1.
    num_inpainting_steps: int
        Number of inpainting steps to run.
    interpolation_steps: int
        Number of interpolation steps to run between inpainting steps.
    num_inference_steps: int
        Number of diffusion steps to run.
    start_step: int
        Step to start diffusion from. The higher the value, the more similar the generated
        image will be to the input image.
    guidance_scale: float
        Guidance scale encourages the model to generate images following the prompt
        closely, albeit at the cost of image quality.
    guidance_rescale: float
        Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    negative_prompt: Optional[PromptType]
        Negative text prompt to uncondition on.
    """

    prompt: PromptType
    image: Image.Image
    walk_type: Union[InpaintWalkType, List[InpaintWalkType]]
    image_height: int = 512
    image_width: int = 512
    height_translation_per_step: int = 64
    width_translation_per_step: int = 64
    translation_factor: Optional[float] = None
    num_inpainting_steps: int = 4
    interpolation_steps: int = 60
    num_inference_steps: int = 50
    start_step: int = 0
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.7
    negative_prompt: Optional[PromptType] = None


class InpaintDiffusion(BaseDiffusion):
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

    def embedding_to_latent(
        self,
        embedding: torch.FloatTensor,
        num_inference_steps: int,
        start_step: int,
        guidance_scale: float,
        guidance_rescale: float,
        latent: torch.FloatTensor,
        mask: torch.FloatTensor,
        masked_image_latent: torch.FloatTensor,
        return_latent_history: bool = False,
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Generate latent by conditioning on prompt embedding and input latent using diffusion.

        Parameters
        ----------
        embedding: torch.FloatTensor
            Embedding of text prompt.
        num_inference_steps: int
            Number of diffusion steps to run.
        start_step: int
            Step to start diffusion from. The higher the value, the more similar the generated
            image will be to the input image.
        guidance_scale: float
            Guidance scale encourages the model to generate images following the prompt
            closely, albeit at the cost of image quality.
        guidance_rescale: float
            Guidance rescale from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        latent: torch.FloatTensor
            Latent to start diffusion from.
        mask: torch.FloatTensor
            Mask to condition on. Values below 0.5 are treated as 0 and values above 0.5
            are treated as 1. Values that are 1 (white) will be inpainted and values that
            are 0 (black) will be preserved.
        masked_image_latent: torch.FloatTensor
            Latent of masked image.
        return_latent_history: bool
            Whether to return latent history. If True, return list of all latents
            generated during diffusion steps.

        Returns
        -------
        Union[torch.FloatTensor, List[torch.FloatTensor]]
            Latent generated by diffusion. If return_latent_history is True, return list of
            all latents generated during diffusion steps.
        """

        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)

        # Add noise to latent based on start step
        start_timestep = (
            self.scheduler.timesteps[start_step].repeat(latent.shape[0]).long()
        )
        noise = self.random_tensor(latent.shape)
        if start_step > 0:
            latent = self.scheduler.add_noise(latent, noise, start_timestep)
        else:
            latent = noise * self.scheduler.init_noise_sigma

        timesteps = self.scheduler.timesteps[start_step:]
        latent_history = [latent]

        # Concatenate mask and masked_image_latent as required by classifier free guidance
        mask = torch.cat([mask] * 2)
        masked_image_latent = torch.cat([masked_image_latent] * 2)

        # Diffusion inference loop
        for i, timestep in tqdm(list(enumerate(timesteps))):
            # Duplicate latent to avoid two forward passes to perform classifier free guidance
            latent_model_input = torch.cat([latent] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            # TODO: The image inpainting model "runwayml/stable-diffusion-inpainting" requires
            # 9 channels as input. The first 4 channels are the image latent, the next 1 channel
            # is the mask and the last 4 channels are the masked image latent. This assumes that
            # all models have the same number of input channels but that may not be the case.
            # Refactor this logic to be more generic.
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latent], dim=1
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

    def _handle_preprocess_tensor(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Helper function to preprocess tensors representing images. Note that
        this should only be used for images already preprocessed with relevant logic in
        the _handle_preprocess_pil and _handle_preprocess_numpy functions.
        """

        if image.ndim != 4 or mask.ndim != 4:
            raise ValueError(
                f"image and mask must have 4 dimensions, got {image.ndim} and {mask.ndim}"
            )
        if image.shape[0] != mask.shape[0]:
            raise ValueError(
                f"image and mask must have same batch size, got {image.shape[0]} and {mask.shape[0]}"
            )
        if image.shape[1] != 3 or mask.shape[1] != 1:
            raise ValueError(
                f"image must have 3 channels and mask must have 1 channel, got {image.shape[1]} and {mask.shape[1]}"
            )
        if image.shape[2] != mask.shape[2] or image.shape[3] != mask.shape[3]:
            raise ValueError(
                f"image and mask must have same height and width, got {image.shape[2:]} and {mask.shape[2:]}"
            )
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                f"image must have values between 0 and 255, got {image.min()} and {image.max()}"
            )
        if mask.min() < 0 or mask.max() > 255:
            raise ValueError(
                f"mask must have values between 0 and 255, got {mask.min()} and {mask.max()}"
            )

        image = (image / 255.0) * 2 - 1
        mask = mask / 255.0

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        image.to(device=self.device, dtype=self.torch_dtype)
        mask.to(device=self.device, dtype=self.torch_dtype)

        masked_image = image * (1 - mask)

        return image, mask, masked_image

    def _handle_preprocess_numpy(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Helper function to preprocess numpy arrays representing images. Note that
        this should only be used for images already preprocessed with relevant logic in
        the _handle_preprocess_pil function.
        """

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=0)
        if image.ndim != 4 or mask.ndim != 4:
            raise ValueError(
                f"image and mask must have 4 dimensions, got {image.ndim} and {mask.ndim}"
            )

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(device=self.device, dtype=self.torch_dtype)

        mask = mask.transpose(0, 3, 1, 2)
        mask = torch.from_numpy(mask).to(device=self.device, dtype=self.torch_dtype)

        return self._handle_preprocess_tensor(image, mask)

    def _handle_preprocess_pil(
        self,
        image: List[Image.Image],
        mask: List[Image.Image],
        image_height: int,
        image_width: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Helper function to preprocess PIL images."""
        image = [
            i.resize((image_height, image_width), resample=Image.LANCZOS).convert("RGB")
            for i in image
        ]
        mask = [
            i.resize((image_height, image_width), resample=Image.LANCZOS).convert("L")
            for i in mask
        ]

        image = np.array(image)
        mask = np.array(mask).reshape(-1, image_height, image_width, 1)

        return self._handle_preprocess_numpy(image, mask)

    @staticmethod
    def _calculate_translation_per_frame(
        translation: int,
        translation_frames: int,
    ) -> List[int]:
        """Helper function to calculate translation per frame."""
        step_size = translation // translation_frames
        remainder = translation % translation_frames
        values = [step_size + (i < remainder) for i in range(translation_frames)]
        return values

    @staticmethod
    def _translate_image_and_mask(
        image: Image.Image,
        walk_type: InpaintWalkType,
        translation: Union[int, Tuple[int, int]],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """Helper function to translate image and mask in given direction by specified translation."""

        def apply_translation(dx: int, dy: int) -> Image.Image:
            return image.transform(
                (image.width, image.height),
                Image.AFFINE,
                (1, 0, dx, 0, 1, dy),
                resample=Image.BICUBIC,
            )

        if walk_type == InpaintWalkType.UP:
            if mask is not None:
                mask[:translation, :] = 255
            new_image = apply_translation(0, -translation)

        elif walk_type == InpaintWalkType.DOWN:
            if mask is not None:
                mask[-translation:, :] = 255
            new_image = apply_translation(0, translation)

        elif walk_type == InpaintWalkType.LEFT:
            if mask is not None:
                mask[:, :translation] = 255
            new_image = apply_translation(-translation, 0)

        elif walk_type == InpaintWalkType.RIGHT:
            if mask is not None:
                mask[:, -translation:] = 255
            new_image = apply_translation(translation, 0)

        elif (
            walk_type == InpaintWalkType.FORWARD
            or walk_type == InpaintWalkType.BACKWARD
        ):
            tw, th = translation

            if mask is not None:
                mask[:th, :] = 255
                mask[-th:, :] = 255
                mask[:, :tw] = 255
                mask[:, -tw:] = 255
            downsampled_image = image.resize(
                (image.width - 2 * tw, image.height - 2 * th), resample=Image.LANCZOS
            )
            new_image = Image.new("RGB", (image.width, image.height))
            new_image.paste(downsampled_image, (tw, th))

        return new_image, Image.fromarray(mask) if mask is not None else None

    @staticmethod
    def _generate_filler_frames(
        start_image: Image.Image,
        end_image: Image.Image,
        walk_type: InpaintWalkType,
        actual_translation: Union[int, Tuple[int, int]],
        filler_translations: Union[int, List[int]],
    ) -> List[Image.Image]:
        """Helper function to generate filler frames for given walk type."""

        if (
            walk_type == InpaintWalkType.FORWARD
            or walk_type == InpaintWalkType.BACKWARD
        ):
            if not isinstance(filler_translations, int):
                raise ValueError(
                    f"filler_translations must be of type int for InpaintWalkType.FORWARD or InpaintWalkType.BACKWARD, got {type(filler_translations)}"
                )
        else:
            if not isinstance(filler_translations, list):
                raise ValueError(
                    f"filler_translations must be of type list for InpaintWalkType.UP, InpaintWalkType.DOWN, InpaintWalkType.LEFT or InpaintWalkType.RIGHT, got {type(filler_translations)}"
                )

        frames = []
        width = start_image.width
        height = start_image.height

        if walk_type == InpaintWalkType.UP:
            for filler_translation in filler_translations:
                a = start_image.crop((0, 0, width, height - filler_translation))
                b = end_image.crop(
                    (
                        0,
                        actual_translation - filler_translation,
                        width,
                        actual_translation,
                    )
                )
                result_img = Image.new("RGB", (width, height))
                result_img.paste(b, (0, 0))
                result_img.paste(a, (0, b.height))
                frames.append(result_img)

        elif walk_type == InpaintWalkType.DOWN:
            for filler_translation in filler_translations:
                a = start_image.crop((0, filler_translation, width, height))
                b = end_image.crop(
                    (
                        0,
                        height - actual_translation,
                        width,
                        height - actual_translation + filler_translation,
                    )
                )
                result_img = Image.new("RGB", (width, height))
                result_img.paste(a, (0, 0))
                result_img.paste(b, (0, a.height))
                frames.append(result_img)

        elif walk_type == InpaintWalkType.LEFT:
            for filler_translation in filler_translations:
                a = start_image.crop((0, 0, width - filler_translation, height))
                b = end_image.crop(
                    (
                        actual_translation - filler_translation,
                        0,
                        actual_translation,
                        height,
                    )
                )
                result_img = Image.new("RGB", (width, height))
                result_img.paste(b, (0, 0))
                result_img.paste(a, (b.width, 0))
                frames.append(result_img)

        elif walk_type == InpaintWalkType.RIGHT:
            for filler_translation in filler_translations:
                a = start_image.crop((filler_translation, 0, width, height))
                b = end_image.crop(
                    (
                        width - actual_translation,
                        0,
                        width - actual_translation + filler_translation,
                        height,
                    )
                )
                result_img = Image.new("RGB", (width, height))
                result_img.paste(a, (0, 0))
                result_img.paste(b, (a.width, 0))
                frames.append(result_img)

        elif (
            walk_type == InpaintWalkType.FORWARD
            or walk_type == InpaintWalkType.BACKWARD
        ):
            aw, ah = actual_translation
            width_factor = 1 - 2 * aw / width
            height_factor = 1 - 2 * ah / height
            width_crop_factor = width - 2 * aw
            height_crop_factor = height - 2 * ah

            translated_image, _ = InpaintDiffusion._translate_image_and_mask(
                start_image, walk_type, actual_translation, mask=None
            )
            translated_image = translated_image.convert("RGBA")
            translated_image = np.array(translated_image)
            translated_image[:ah, :, 3] = 0
            translated_image[-ah:, :, 3] = 0
            translated_image[:, :aw, 3] = 0
            translated_image[:, -aw:, 3] = 0
            translated_image = Image.fromarray(translated_image)

            end_image.paste(translated_image, mask=translated_image)

            for i in range(filler_translations - 1):
                translation_factor = 1 - (i + 1) / filler_translations
                interpolation_image = end_image
                interpolation_width = round(
                    (1 - width_factor**translation_factor) * width / 2
                )
                interpolation_height = round(
                    (1 - height_factor**translation_factor) * height / 2
                )

                interpolation_image = interpolation_image.crop(
                    (
                        interpolation_width,
                        interpolation_height,
                        width - interpolation_width,
                        height - interpolation_height,
                    )
                ).resize((width, height), resample=Image.LANCZOS)

                w = width - 2 * interpolation_width
                h = height - 2 * interpolation_height
                crop_fix_width = round((1 - width_crop_factor / w) * width / 2)
                crop_fix_height = round((1 - height_crop_factor / h) * height / 2)

                start_image_crop_fix, _ = InpaintDiffusion._translate_image_and_mask(
                    start_image, walk_type, (crop_fix_width, crop_fix_height), mask=None
                )
                start_image_crop_fix = start_image_crop_fix.convert("RGBA")
                start_image_crop_fix = np.array(start_image_crop_fix)
                start_image_crop_fix[:crop_fix_height, :, 3] = 0
                start_image_crop_fix[-crop_fix_height:, :, 3] = 0
                start_image_crop_fix[:, :crop_fix_width, 3] = 0
                start_image_crop_fix[:, -crop_fix_width:, 3] = 0
                start_image_crop_fix = Image.fromarray(start_image_crop_fix)

                interpolation_image.paste(
                    start_image_crop_fix, mask=start_image_crop_fix
                )
                frames.append(interpolation_image)

            frames.append(end_image)

        return frames

    def preprocess_image_and_mask(
        self,
        image: ImageType,
        mask: ImageType,
        image_height: int,
        image_width: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Preprocess image and mask to be used for inference.

        Parameters
        ----------
        image: ImageType
            Input image to condition on.
        mask: ImageType
            Input mask to condition on. Must have same height and width as image.
            Must have 1 channel. Must have values between 0 and 1. Values below 0.5
            are treated as 0 and values above 0.5 are treated as 1. Values that are
            1 (white) will be inpainted and values that are 0 (black) will be
            preserved.
        image_height: int
            Height of image to generate. If height of provided image and image_height
            do not match, image will be resized to image_height using PIL Lanczos method.
        image_width: int
            Width of image to generate. If width of provided image and image_width
            do not match, image will be resized to image_width using PIL Lanczos method.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Tuple of image, mask and masked_image.
        """

        if image is None or mask is None:
            raise ValueError("image and mask cannot be None")
        if type(image) != type(mask):
            raise TypeError(
                f"image and mask must be of same type, got {type(image)} and {type(mask)}"
            )
        if isinstance(image, Image.Image):
            image = [image]
            mask = [mask]

        if isinstance(image, torch.Tensor):
            image, mask, masked_image = self._handle_preprocess_tensor(image, mask)
        elif isinstance(image, np.ndarray):
            image, mask, masked_image = self._handle_preprocess_numpy(image, mask)
        else:
            image, mask, masked_image = self._handle_preprocess_pil(
                image, mask, image_height, image_width
            )

        return image, mask, masked_image

    @torch.no_grad()
    def __call__(
        self,
        config: InpaintConfig,
    ) -> OutputType:
        """
        Run inference by conditioning on input image, mask and prompt.

        Parameters
        ----------
        config: InpaintConfig
            Configuration for running inference with InpaintDiffusion.

        Returns
        -------
        OutputType
            Generated output based on output_type.
        """

        prompt = config.prompt
        image = config.image
        mask = config.mask
        image_height = config.image_height
        image_width = config.image_width
        num_inference_steps = config.num_inference_steps
        start_step = config.start_step
        guidance_scale = config.guidance_scale
        guidance_rescale = config.guidance_rescale
        negative_prompt = config.negative_prompt
        output_type = config.output_type
        return_latent_history = config.return_latent_history

        # Validate input
        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_height=image_height,
            image_width=image_width,
            start_step=start_step,
            num_inference_steps=num_inference_steps,
        )

        # Preprocess image and mask
        image, mask, masked_image = self.preprocess_image_and_mask(
            image=image,
            mask=mask,
            image_height=image_height,
            image_width=image_width,
        )

        # Generate latent from input image and masked_image. Mask is down/up-scaled to match latent shape
        image_latent = self.image_to_latent(image)
        mask = F.interpolate(mask, size=(image_latent.shape[2], image_latent.shape[3]))
        masked_image_latent = self.image_to_latent(masked_image)

        # Generate embedding to condition on prompt and uncondition on negative prompt
        embedding = self.prompt_to_embedding(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Run inference
        latent = self.embedding_to_latent(
            embedding=embedding,
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            latent=image_latent,
            mask=mask,
            masked_image_latent=masked_image_latent,
            return_latent_history=return_latent_history,
        )

        return self.resolve_output(
            latent=latent,
            output_type=output_type,
            return_latent_history=return_latent_history,
        )

    generate = __call__

    def walk(
        self,
        config: InpaintWalkConfig,
    ) -> OutputType:
        """
        Inpaint image by walking in direction(s) of choice.

        Parameters
        ----------
        config: InpaintWalkConfig
            Configuration for running inpaint walking with InpaintDiffusion.

        Returns
        -------
        OutputType
            Generated output based on output_type.
        """

        prompt = config.prompt
        image = config.image
        walk_type = config.walk_type
        image_height = config.image_height
        image_width = config.image_width
        height_translation_per_step = config.height_translation_per_step
        width_translation_per_step = config.width_translation_per_step
        translation_factor = config.translation_factor
        num_inpainting_steps = config.num_inpainting_steps
        interpolation_steps = config.interpolation_steps
        num_inference_steps = config.num_inference_steps
        start_step = config.start_step
        guidance_scale = config.guidance_scale
        guidance_rescale = config.guidance_rescale
        negative_prompt = config.negative_prompt

        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_height=image_height,
            image_width=image_width,
            num_inference_steps=num_inference_steps,
        )

        # TODO: Make validate input handle this
        if translation_factor is not None:
            if translation_factor < 0 or translation_factor > 1:
                raise ValueError(
                    f"translation_factor must be between 0 and 1, got {translation_factor}"
                )
            height_translation_per_step = int(image_height * translation_factor)
            width_translation_per_step = int(image_width * translation_factor)

        if isinstance(walk_type, InpaintWalkType):
            walk_type = [walk_type]
        if not isinstance(walk_type, list):
            raise TypeError(
                f"walk_type must be of type InpaintWalkType or List[InpaintWalkType], got {type(walk_type)}"
            )

        if len(walk_type) == 1:
            walk_type = walk_type * num_inpainting_steps
        if len(walk_type) != num_inpainting_steps:
            raise ValueError(
                f"walk_type must have length of num_inpainting_steps, got {len(walk_type)} and {num_inpainting_steps}"
            )

        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            if len(prompt) == 1:
                prompt = prompt * num_inpainting_steps
            if len(prompt) != num_inpainting_steps:
                raise ValueError(
                    f"prompt must have length of num_inpainting_steps, got {len(walk_type)} and {num_inpainting_steps}"
                )

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            if len(negative_prompt) == 1:
                negative_prompt = negative_prompt * num_inpainting_steps
            if len(negative_prompt) != num_inpainting_steps:
                raise ValueError(
                    f"negative_prompt must have length of num_inpainting_steps, got {len(walk_type)} and {num_inpainting_steps}"
                )

        walk_has_backward = InpaintWalkType.BACKWARD in walk_type
        walk_has_forward = InpaintWalkType.FORWARD in walk_type
        if walk_has_backward or walk_has_forward:
            if height_translation_per_step * 2 > image_height:
                raise ValueError(
                    f"height_translation_per_step must be less than half of image_height, got {height_translation_per_step} and {image_height}"
                )
            if width_translation_per_step * 2 > image_width:
                raise ValueError(
                    f"width_translation_per_step must be less than half of image_width, got {width_translation_per_step} and {image_width}"
                )
        else:
            if height_translation_per_step >= image_height:
                raise ValueError(
                    f"height_translation_per_step must be less than image_height, got {height_translation_per_step} and {image_height}"
                )
            if width_translation_per_step >= image_width:
                raise ValueError(
                    f"width_translation_per_step must be less than image_width, got {width_translation_per_step} and {image_width}"
                )

        height_filler_translations = self._calculate_translation_per_frame(
            translation=height_translation_per_step,
            translation_frames=interpolation_steps,
        )
        width_filler_translations = self._calculate_translation_per_frame(
            translation=width_translation_per_step,
            translation_frames=interpolation_steps,
        )

        for i in range(1, interpolation_steps):
            height_filler_translations[i] += height_filler_translations[i - 1]
            width_filler_translations[i] += width_filler_translations[i - 1]

        assert height_filler_translations[-1] == height_translation_per_step
        assert width_filler_translations[-1] == width_translation_per_step

        image = image.resize(
            (image_height, image_width), resample=Image.LANCZOS
        ).convert("RGB")

        base_mask = np.zeros((image_width, image_height), dtype=np.uint8)
        prev_image = image
        frames = []

        for prompt, negative_prompt, walk in tqdm(
            zip(prompt, negative_prompt, walk_type)
        ):
            if walk == InpaintWalkType.LEFT or walk == InpaintWalkType.RIGHT:
                translation = width_translation_per_step
                filler_translations = width_filler_translations
            elif walk == InpaintWalkType.UP or walk == InpaintWalkType.DOWN:
                translation = height_translation_per_step
                filler_translations = height_filler_translations
            elif walk == InpaintWalkType.BACKWARD:
                translation = (width_translation_per_step, height_translation_per_step)
                filler_translations = interpolation_steps
            elif walk == InpaintWalkType.FORWARD:
                raise ValueError(
                    "InpaintWalkType.FORWARD is not supported yet. If you would like to do this, reverse the sequence of images generated using InpaintWalkType.BACKWARD"
                )
            else:
                raise ValueError(f"Invalid Inpaint Walk Type: {walk}")

            image, mask = self._translate_image_and_mask(
                prev_image, walk, translation, mask=base_mask.copy()
            )
            generated_image = self.generate(
                prompt=prompt,
                image=image,
                mask=mask,
                image_height=image_height,
                image_width=image_width,
                num_inference_steps=num_inference_steps,
                start_step=start_step,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                negative_prompt=negative_prompt,
                output_type="pil",
                return_latent_history=False,
            )[0]

            filler_frames_list = self._generate_filler_frames(
                start_image=prev_image,
                end_image=generated_image,
                walk_type=walk,
                actual_translation=translation,
                filler_translations=filler_translations,
            )

            prev_image = filler_frames_list[-1].copy()

            if walk == InpaintWalkType.FORWARD:
                filler_frames_list = filler_frames_list[::-1]

            frames.extend(filler_frames_list)

        return frames
