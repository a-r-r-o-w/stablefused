import torch

from dataclasses import dataclass
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    SchedulerMixin,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, Dict


@dataclass
class StableDiffusionModel:
    """A dataclass to hold components of a StableDiffusion model"""

    model_id: str
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    scheduler: SchedulerMixin


class ModelCache:
    """
    A cache for StableDiffusion models. This class should not be instantiated by the user.
    You should use the load_model function instead. It is a mapping from model_id to
    StableDiffusionModel. This allows us to avoid loading the same model components multiple
    times.
    """

    def __init__(self) -> None:
        self.cache = dict()

    def __getitem__(
        self,
        model_id: str,
        torch_dtype: torch.dtype,
    ) -> StableDiffusionModel:
        if model_id not in self.cache:
            self.cache[model_id] = StableDiffusionModel(
                model_id=model_id,
                tokenizer=CLIPTokenizer.from_pretrained(
                    model_id, subfolder="tokenizer"
                ),
                text_encoder=CLIPTextModel.from_pretrained(
                    model_id, subfolder="text_encoder", torch_dtype=torch_dtype
                ),
                vae=AutoencoderKL.from_pretrained(
                    model_id, subfolder="vae", torch_dtype=torch_dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(
                    model_id, subfolder="unet", torch_dtype=torch_dtype
                ),
                scheduler=DPMSolverMultistepScheduler.from_pretrained(
                    model_id, subfolder="scheduler"
                ),
            )
        return self.cache[model_id]

    get = __getitem__

    def set(
        self,
        name: str,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ) -> StableDiffusionModel:
        self.cache[name] = StableDiffusionModel(
            model_id=name,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        return self.cache[name]


_model_cache = ModelCache()


def load_model(
    model_id: str, torch_dtype: torch.dtype = torch.float32
) -> StableDiffusionModel:
    return _model_cache.get(model_id, torch_dtype)


def cache_model(
    name: str,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: KarrasDiffusionSchedulers,
) -> StableDiffusionModel:
    return _model_cache.set(name, tokenizer, text_encoder, vae, unet, scheduler)
