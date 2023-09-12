import numpy as np
import torch

from diffusers.schedulers import (
    DEISMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverSDEScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from typing import Any, Dict, Union
from stablefused.typing import Scheduler, SchedulerType


def lerp(
    v0: Union[torch.Tensor, np.ndarray],
    v1: Union[torch.Tensor, np.ndarray],
    t: Union[float, torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Linearly interpolate between two vectors/tensors.

    Parameters
    ----------
    v0: Union[torch.Tensor, np.ndarray]
        First vector/tensor.
    v1: Union[torch.Tensor, np.ndarray]
        Second vector/tensor.
    t: Union[float, torch.Tensor, np.ndarray]
        Interpolation factor. If float, must be between 0 and 1. If np.ndarray or
        torch.Tensor, must be one dimensional with values between 0 and 1.

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Interpolated vector/tensor between v0 and v1.
    """
    inputs_are_torch = False
    t_is_float = False

    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
    if isinstance(v1, torch.Tensor):
        inputs_are_torch = True
        input_device = v1.device
        v1 = v1.cpu().numpy()
    if isinstance(t, torch.Tensor):
        inputs_are_torch = True
        input_device = t.device
        t = t.cpu().numpy()
    elif isinstance(t, float):
        t_is_float = True
        t = np.array([t])

    t = t[..., None]
    v0 = v0[None, ...]
    v1 = v1[None, ...]
    v2 = (1 - t) * v0 + t * v1

    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def slerp(
    v0: Union[torch.Tensor, np.ndarray],
    v1: Union[torch.Tensor, np.ndarray],
    t: Union[float, torch.Tensor, np.ndarray],
    DOT_THRESHOLD=0.9995,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Spherical linear interpolation between two vectors/tensors.

    Parameters
    ----------
    v0: Union[torch.Tensor, np.ndarray]
        First vector/tensor.
    v1: Union[torch.Tensor, np.ndarray]
        Second vector/tensor.
    t: Union[float, np.ndarray]
        Interpolation factor. If float, must be between 0 and 1. If np.ndarray, must be one
        dimensional with values between 0 and 1.
    DOT_THRESHOLD: float
        Threshold for when to use linear interpolation instead of spherical interpolation.

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Interpolated vector/tensor between v0 and v1.
    """
    inputs_are_torch = False
    t_is_float = False

    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
    if isinstance(v1, torch.Tensor):
        inputs_are_torch = True
        input_device = v1.device
        v1 = v1.cpu().numpy()
    if isinstance(t, torch.Tensor):
        inputs_are_torch = True
        input_device = t.device
        t = t.cpu().numpy()
    elif isinstance(t, float):
        t_is_float = True
        t = np.array([t], dtype=v0.dtype)

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        # v1 and v2 are close to parallel
        # Use linear interpolation instead
        v2 = lerp(v0, v1, t)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        s0 = s0[..., None]
        s1 = s1[..., None]
        v0 = v0[None, ...]
        v1 = v1[None, ...]
        v2 = s0 * v0 + s1 * v1

    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def resolve_scheduler(
    scheduler_type: Scheduler, config: Dict[str, Any]
) -> SchedulerType:
    if scheduler_type == Scheduler.DEIS:
        return DEISMultistepScheduler.from_config(config)

    elif scheduler_type == Scheduler.DDIM:
        return DDIMScheduler.from_config(config)

    elif scheduler_type == Scheduler.DDPM:
        return DDPMScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM2_KARRAS:
        return KDPM2DiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM2_KARRAS_ANCESTRAL:
        return KDPM2AncestralDiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM_SDE:
        return DPMSolverSDEScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM_SDE_KARRAS:
        return DPMSolverSDEScheduler.from_config(config, use_karras_sigmas=True)

    elif scheduler_type == Scheduler.DPM_MULTISTEP:
        return DPMSolverMultistepScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM_MULTISTEP_KARRAS:
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

    elif scheduler_type == Scheduler.DPM_SINGLESTEP:
        return DPMSolverSinglestepScheduler.from_config(config)

    elif scheduler_type == Scheduler.DPM_SINGLESTEP_KARRAS:
        return DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=True)

    elif scheduler_type == Scheduler.EULER:
        return EulerDiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.EULER_ANCESTRAL:
        return EulerAncestralDiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.HEUN:
        return HeunDiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.LINEAR_MULTISTEP:
        return LMSDiscreteScheduler.from_config(config)

    elif scheduler_type == Scheduler.PNDM:
        return PNDMScheduler.from_config(config)

    elif scheduler_type == Scheduler.UNIPC:
        return UniPCMultistepScheduler.from_config(config)
