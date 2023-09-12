import imageio
import numpy as np
import torch

from PIL import Image
from typing import List, Union


def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert pytorch tensor to numpy image.

    Parameters
    ----------
    images: torch.FloatTensor
        Image represented as a pytorch tensor (N, C, H, W).

    Returns
    -------
    np.ndarray
        Image represented as a numpy array (N, H, W, C).
    """
    return images.detach().cpu().permute(0, 2, 3, 1).float().numpy()


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert numpy image to pytorch tensor.

    Parameters
    ----------
    images: np.ndarray
        Image represented as a numpy array (N, H, W, C).

    Returns
    -------
    torch.FloatTensor
        Image represented as a pytorch tensor (N, C, H, W).
    """
    if images.ndim == 3:
        images = images[..., None]
    return torch.from_numpy(images.transpose(0, 3, 1, 2))


def numpy_to_pil(images: np.ndarray) -> Image.Image:
    """
    Convert numpy image to PIL image.

    Parameters
    ----------
    images: np.ndarray
        Image represented as a numpy array (N, H, W, C).

    Returns
    -------
    Image.Image
        Image represented as a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    """
    Convert PIL image to numpy image.

    Parameters
    ----------
    images: Union[List[Image.Image], Image.Image]
        PIL image or list of PIL images.

    Returns
    -------
    np.ndarray
        Image represented as a numpy array (N, H, W, C).
    """
    if not isinstance(images, Image.Image) and not isinstance(images, list):
        raise ValueError(
            f"Expected PIL image or list of PIL images, got {type(images)}."
        )
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)
    return images


def normalize(images: torch.FloatTensor) -> torch.FloatTensor:
    """
    Normalize an image array to the range [-1, 1].

    Parameters
    ----------
    images: torch.FloatTensor
        Image represented as a pytorch tensor (N, C, H, W).

    Returns
    -------
    torch.FloatTensor
        Normalized image as pytorch tensor.
    """
    return 2.0 * images - 1.0


def denormalize(images: torch.FloatTensor) -> torch.FloatTensor:
    """
    Denormalize an image array to the range [0.0, 1.0].

    Parameters
    ----------
    images: torch.FloatTensor
        Image represented as a pytorch tensor (N, C, H, W).

    Returns
    -------
    torch.FloatTensor
        Denormalized image as pytorch tensor.
    """
    return (0.5 + images / 2).clamp(0, 1)


def pil_to_video(images: List[Image.Image], filename: str, fps: int = 60) -> None:
    """
    Convert a list of PIL images to a video.

    Parameters
    ----------
    images: List[Image.Image]
        List of PIL images.
    filename: str
        Filename to save video to.
    fps: int
        Frames per second of video.
    """
    frames = [np.array(image) for image in images]
    with imageio.get_writer(filename, fps=fps) as video_writer:
        for frame in frames:
            video_writer.append_data(frame)


def pil_to_gif(images: List[Image.Image], filename: str, fps: int = 60) -> None:
    """
    Convert a list of PIL images to a GIF.

    Parameters
    ----------
    images: List[Image.Image]
        List of PIL images.
    filename: str
        Filename to save GIF to.
    fps: int
        Frames per second of GIF.
    """
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )


def image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """
    Create a grid of images on a single PIL image.

    Parameters
    ----------
    images: List[Image.Image]
        List of PIL images.
    rows: int
        Number of rows in grid.
    cols: int
        Number of columns in grid.

    Returns
    -------
    Image.Image
        Grid of images as a PIL image.
    """
    if len(images) > rows * cols:
        raise ValueError(
            f"Number of images ({len(images)}) exceeds grid size ({rows}x{cols})."
        )
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid
