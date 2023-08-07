import numpy as np
import torch

from PIL import Image
from typing import List, Union


def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """Convert pytorch tensor to numpy image."""
    return images.detach().cpu().permute(0, 2, 3, 1).float().numpy()


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert numpy image to pytorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    return torch.from_numpy(images.transpose(0, 3, 1, 2))


def numpy_to_pil(images: np.ndarray) -> Image.Image:
    """Convert numpy image to PIL image."""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    """Convert PIL image to numpy image."""
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)
    return images


def normalize(images: torch.FloatTensor):
    """Normalize an image array to the range [-1, 1]."""
    return 2.0 * images - 1.0


def denormalize(images: torch.FloatTensor):
    """Denormalize an image array to the range [0.0, 1.0]"""
    return (0.5 + images / 2).clamp(0, 1)
