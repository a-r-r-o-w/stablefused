import numpy as np
import torch

from PIL import Image
from diffusers.models import UNet2DConditionModel, UNet3DConditionModel
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from typing import List, Union


ImageType = Union[torch.Tensor, np.ndarray, Image.Image, List[Image.Image]]

OutputType = Union[torch.Tensor, np.ndarray, List[Image.Image]]

PromptType = Union[str, List[str]]

SchedulerType = Union[
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
]

UNetType = Union[UNet2DConditionModel, UNet3DConditionModel]
