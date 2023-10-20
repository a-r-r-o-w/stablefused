import numpy as np
import torch
import pytest

from stablefused import (
    LatentWalkConfig,
    LatentWalkInterpolateConfig,
    LatentWalkDiffusion,
)


@pytest.fixture
def model():
    """
    Fixture to initialize the LatentWalkDiffusion model and set random seeds for reproducibility.

    Returns
    -------
    LatentWalkDiffusion
        The initialized LatentWalkDiffusion model.
    """
    seed = 1337
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LatentWalkDiffusion(model_id=model_id, device=device)
    return model


@pytest.fixture
def config():
    return {
        "prompt": "a photo of a cat",
        "num_inference_steps": 1,
        "image_dim": 32,
    }


@pytest.fixture
def config_interpolate():
    return {
        "prompt": ["a photo of a cat", "a photo of a dog"],
        "num_inference_steps": 1,
        "interpolation_steps": 5,
        "image_dim": 32,
    }


def test_latent_walk_diffusion(model: LatentWalkDiffusion, config: dict) -> None:
    """
    Test case to check if the LatentWalkDiffusion is working correctly.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """

    dim = config.get("image_dim")
    image = model.random_tensor((1, 3, dim, dim))
    latent = model.image_to_latent(image)

    images = model(
        LatentWalkConfig(
            prompt=config.get("prompt"),
            latent=latent,
            num_inference_steps=config.get("num_inference_steps"),
            output_type="np",
        )
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


def test_interpolate(model: LatentWalkDiffusion, config_interpolate: dict) -> None:
    """
    Test case to check if the LatentWalkDiffusion is working correctly.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """

    dim = config_interpolate.get("image_dim")
    num_prompts = len(config_interpolate.get("prompt"))
    image_count = config_interpolate.get("interpolation_steps") * (num_prompts - 1)
    image = model.random_tensor((num_prompts, 3, dim, dim))
    latent = model.image_to_latent(image)

    images = model.interpolate(
        LatentWalkInterpolateConfig(
            prompt=config_interpolate.get("prompt"),
            latent=latent,
            num_inference_steps=config_interpolate.get("num_inference_steps"),
            interpolation_steps=config_interpolate.get("interpolation_steps"),
            output_type="np",
        )
    )

    assert type(images) is np.ndarray
    assert images.shape == (image_count, dim, dim, 3)


if __name__ == "__main__":
    pytest.main([__file__])
