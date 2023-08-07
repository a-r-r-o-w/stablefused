import numpy as np
import torch
import pytest

from stablefused import TextToImageDiffusion


@pytest.fixture
def model():
    """
    Fixture to initialize the TextToImageDiffusion model and set random seeds for reproducibility.

    Returns
    -------
    TextToImageDiffusion
        The initialized TextToImageDiffusion model.
    """
    seed = 1337
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TextToImageDiffusion(model_id=model_id, device=device)
    return model


@pytest.fixture
def config():
    return {
        "prompt": "a photo of a cat",
        "num_inference_steps": 1,
        "image_dim": 64,
    }


def test_text_to_image_diffusion(model: TextToImageDiffusion, config: dict) -> None:
    """
    Test case for the TextToImageDiffusion model.

    Parameters
    ----------
    model
        The initialized TextToImageDiffusion model fixture.
    config
        The configuration dictionary for the test case.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """

    dim = config.get("image_dim")
    images = model(
        prompt=config.get("prompt"),
        image_height=dim,
        image_width=dim,
        num_inference_steps=config.get("num_inference_steps"),
        output_type="np",
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


def test_return_latent_history(model: TextToImageDiffusion, config: dict) -> None:
    """
    Test case for the TextToImageDiffusion model.

    Parameters
    ----------
    model
        The initialized TextToImageDiffusion model fixture.
    config
        The configuration dictionary for the test case.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """

    dim = config.get("image_dim")
    images = model(
        prompt=config.get("prompt"),
        image_height=dim,
        image_width=dim,
        num_inference_steps=config.get("num_inference_steps"),
        output_type="pt",
        return_latent_history=True,
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, config.get("num_inference_steps") + 1, 3, dim, dim)


if __name__ == "__main__":
    pytest.main([__file__])
