import numpy as np
import torch
import pytest

from stablefused import LatentWalkDiffusion


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
    image = torch.randn(1, 3, dim, dim)
    latent = model.image_to_latent(image)

    images = model(
        prompt=config.get("prompt"),
        latent=latent,
        num_inference_steps=config.get("num_inference_steps"),
        output_type="np",
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


def test_no_classifier_free_guidance(model: LatentWalkDiffusion, config: dict) -> None:
    """
    Test case to check if the LatentWalkDiffusion is working correctly when classifier
    free guidance is disabled.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = torch.randn(1, 3, dim, dim)
    latent = model.image_to_latent(image)

    images = model(
        prompt=config.get("prompt"),
        latent=latent,
        num_inference_steps=config.get("num_inference_steps"),
        output_type="np",
        guidance_scale=1.0,  # setting guidance_scale <= 1.0 effectively disables classifier free guidance
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


if __name__ == "__main__":
    pytest.main([__file__])
