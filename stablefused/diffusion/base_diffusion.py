import numpy as np
import torch

from PIL import Image
from abc import ABC, abstractmethod
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, List, Optional, Tuple, Union

from stablefused.typing import (
    PromptType,
    OutputType,
    Scheduler,
    SchedulerType,
    UNetType,
)
from stablefused.utils import (
    denormalize,
    load_model_from_cache,
    normalize,
    numpy_to_pil,
    numpy_to_pt,
    pil_to_numpy,
    pt_to_numpy,
    resolve_scheduler,
    save_model_to_cache,
)


class BaseDiffusion(ABC):
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
        use_cache=True,
        *args,
        **kwargs,
    ) -> None:
        self.device: str = device
        self.torch_dtype: torch.dtype = torch_dtype
        self.model_id: str = model_id

        self.tokenizer: CLIPTokenizer
        self.text_encoder: CLIPTextModel
        self.vae: AutoencoderKL
        self.unet: UNetType
        self.scheduler: SchedulerType
        self.vae_scale_factor: int

        if model_id is None:
            if (
                tokenizer is None
                or text_encoder is None
                or vae is None
                or unet is None
                or scheduler is None
            ):
                raise ValueError(
                    "Either (`model_id`) or (`tokenizer`, `text_encoder`, `vae`, `unet`, `scheduler`) must be provided."
                )

            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
            self.vae = vae
            self.unet = unet
            self.scheduler = scheduler
        else:
            model = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, *args, **kwargs
            )
            self.tokenizer = model.tokenizer
            self.text_encoder = model.text_encoder
            self.vae = model.vae
            self.unet = model.unet
            self.scheduler = model.scheduler

        self.to(self.device)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if use_cache and model_id is not None:
            model = load_model_from_cache(model_id, None)
            if model is None:
                save_model_to_cache(self)
            else:
                self.share_components_with(model)

    def to(self, device: str) -> "BaseDiffusion":
        """
        Move model to specified compute device.

        Parameters
        ----------
        device: str
            The device to move the model to. Must be one of `cuda` or `cpu`.
        """
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        return self

    def share_components_with(self, model: "BaseDiffusion") -> None:
        """
        Share components with another model. This allows for sharing of the
        different internal components of the model, such as the text encoder,
        VAE, and UNet. This is useful for reducing memory usage when using
        multiple diffusion pipelines with the same checkpoint at the same time.

        Parameters
        ----------
        model: BaseDiffusion
            The model to share components with.
        """
        self.device = model.device
        self.torch_dtype = model.torch_dtype
        self.tokenizer = model.tokenizer
        self.text_encoder = model.text_encoder
        self.vae = model.vae
        self.unet = model.unet
        self.scheduler = model.scheduler
        self.vae_scale_factor = model.vae_scale_factor

    def set_scheduler(self, scheduler: Scheduler) -> None:
        """
        Set the scheduler for the diffusion pipeline.

        Parameters
        ----------
        scheduler: SchedulerType
            The scheduler to use for the diffusion pipeline.
        """
        self.scheduler = resolve_scheduler(scheduler, self.scheduler.config)

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
        prompt: PromptType = None,
        negative_prompt: PromptType = None,
        image_height: int = None,
        image_width: int = None,
        start_step: int = None,
        num_inference_steps: int = None,
        strength: float = None,
    ) -> None:
        """
        Validate input parameters.
        TODO: This needs to be removed and improved. More checks need to be added.

        Parameters
        ----------
        prompt: PromptType
            The prompt(s) to condition on.
        negative_prompt: PromptType
            The negative prompt(s) to condition on.
        image_height: int
            The height of the image to generate.
        image_width: int
            The width of the image to generate.
        start_step: int
            The step to start inference from.
        num_inference_steps: int
            The number of inference steps to perform.
        strength: float
            The strength of the noise mixing when performing LatentWalkDiffusion.
        """
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
        """
        Abstract method for converting an embedding to a latent vector. This method
        must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for performing inference. This method must be implemented
        by all subclasses.
        """
        pass

    def random_tensor(self, shape: Union[List[int], Tuple[int]]) -> torch.FloatTensor:
        """
        Generate a random tensor of the specified shape.

        Parameters
        ----------
        shape: List[int] or Tuple[int]
            The shape of the random tensor to generate.

        Returns
        -------
        torch.FloatTensor
            A random tensor of the specified shape on the same device and dtype
            as model.
        """
        rand_tensor = torch.randn(shape, device=self.device, dtype=self.torch_dtype)
        return rand_tensor

    def prompt_to_embedding(
        self,
        prompt: PromptType,
        negative_prompt: Optional[PromptType] = None,
    ) -> torch.FloatTensor:
        """
        Convert a prompt or a list of prompts into a text embedding.

        Parameters
        ----------
        prompt: PromptType
            The prompt or a list of prompts to convert into an embedding. Used
            for conditioning.
        negative_prompt: Optional[PromptType]
            A negative prompt or a list of negative prompts, by default None.
            Use for unconditioning. If not provided, an empty string ('') will
            be used to generate the unconditional embeddings.

        Returns
        -------
        torch.FloatTensor
            A text embedding generated from the given prompt(s) and, if provided,
            the negative prompt(s).
        """

        if negative_prompt is not None and type(negative_prompt) is not type(prompt):
            raise TypeError(
                f"`negative_prompt` must have the same type as `prompt` ({type(prompt)}), but found {type(negative_prompt)}"
            )

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if negative_prompt is not None:
                negative_prompt = [negative_prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise TypeError("`prompt` must be a string or a list of strings")

        # Tokenize the prompt(s)
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Enable use of attention_mask if the text_encoder supports it
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        # Generate text embedding
        text_embedding = self.text_encoder(
            text_input.input_ids.to(self.device), attention_mask=attention_mask
        )[0]

        # Unconditioning input is an empty string if negative_prompt is not provided
        if negative_prompt is None:
            unconditioning_input = [""] * batch_size
        else:
            unconditioning_input = negative_prompt

        # Tokenize the unconditioning input
        unconditioning_input = self.tokenizer(
            unconditioning_input,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Generate unconditional embedding
        unconditional_embedding = self.text_encoder(
            unconditioning_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )[0]

        # Concatenate unconditional and conditional embeddings
        embedding = torch.cat([unconditional_embedding, text_embedding])
        return embedding

    def classifier_free_guidance(
        self,
        noise_prediction: torch.FloatTensor,
        guidance_scale: float,
        guidance_rescale: float,
    ) -> torch.FloatTensor:
        """
        Apply classifier-free guidance to noise prediction.

        Parameters
        ----------
        noise_prediction: torch.FloatTensor
            The noise prediction tensor to which guidance will be applied.
        guidance_scale: float
            The scale factor for applying guidance to the noise prediction.
        guidance_rescale: float
            The rescale factor for adjusting the noise prediction based on
            guidance. Based on findings in Section 3.4  of [Common Diffusion
            Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).

        Returns
        -------
        torch.FloatTensor
            The noise prediction tensor after applying classifier-free guidance.
        """

        # Perform guidance
        noise_unconditional, noise_prompt = noise_prediction.chunk(2)
        noise_prediction = noise_unconditional + guidance_scale * (
            noise_prompt - noise_unconditional
        )

        # Skip computing std if guidance_rescale is 0
        if guidance_rescale > 0:
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
    ) -> OutputType:
        """
        Convert a latent tensor to an image in the specified output format.

        Parameters
        ----------
        latent: torch.FloatTensor
            The latent tensor to convert into an image.
        output_type: str
            The desired output format for the image. Should be one of [`pt`, `np`, `pil`].

        Returns
        -------
        OutputType
            An image in the specified output format.
        """
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
        image: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor],
    ) -> torch.FloatTensor:
        """
        Convert an image or a list of images into a latent tensor.

        Parameters
        ----------
        image: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor]
            The input image(s) to convert into a latent tensor. Supported types are
            `PIL.Image.Image`, `np.ndarray`, and `torch.Tensor`.

        Returns
        -------
        torch.FloatTensor
            A latent tensor representing the input image(s).
        """
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
    ) -> Union[OutputType, List[OutputType]]:
        """
        Resolve the output from the latent based on the provided output options.

        Parameters
        ----------
        latent: torch.FloatTensor
            The latent tensor representing the content to be resolved.
        output_type: str
            The desired output format. Should be one of [`latent`, `pt`, `np`, `pil`].
        return_latent_history: bool
            If True, it means that the input latent tensor contains a tensor of latent
            tensor for each inference step. This requires decoding each latent tensor
            and returning a list of images. If False, decoding occurs directly.

        Returns
        -------
        Union[OutputType, List[OutputType]]
            The resolved output based on the provided latent vector and options.
        """
        if output_type not in ["latent", "pt", "np", "pil"]:
            raise ValueError(
                "`output_type` must be one of [`latent`, `pt`, `np`, `pil`]"
            )

        if output_type == "latent":
            return latent

        if return_latent_history:
            # Transpose latent tensor from [num_steps, batch_size, *latent_dim] to
            # [batch_size, num_steps, *latent_dim].
            # This is done so that the history of latent vectors for each prompt
            # is returned as a row instead of a column. It is what the user would
            # intuitively expect.
            latent = torch.transpose(latent, 0, 1)
            image = [
                self.latent_to_image(l, output_type)
                for _, l in list(enumerate(tqdm(latent)))
            ]

            if output_type == "pt":
                image = torch.stack(image)
            elif output_type == "np":
                image = np.stack(image)
            else:
                # output type is "pil" so we can just return as a python list
                pass
        else:
            image = self.latent_to_image(latent, output_type)

        return image
