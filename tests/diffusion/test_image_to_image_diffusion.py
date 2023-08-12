import numpy as np
import torch
import pytest

from stablefused import ImageToImageDiffusion


@pytest.fixture
def model():
    """
    Fixture to initialize the ImageToImageDiffusion model and set random seeds for reproducibility.

    Returns
    -------
    ImageToImageDiffusion
        The initialized ImageToImageDiffusion model.
    """
    seed = 1337
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ImageToImageDiffusion(model_id=model_id, device=device)
    return model


@pytest.fixture
def config():
    return {
        "prompt": "a photo of a cat",
        "num_inference_steps": 2,
        "start_step": 1,
        "image_dim": 32,
    }


def test_image_to_image_diffusion(model: ImageToImageDiffusion, config: dict) -> None:
    """
    Test case to check if the ImageToImageDiffusion is working correctly.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = model.random_tensor((1, 3, dim, dim))

    images = model(
        image=image,
        prompt=config.get("prompt"),
        num_inference_steps=config.get("num_inference_steps"),
        output_type="np",
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


def test_return_latent_history(model: ImageToImageDiffusion, config: dict) -> None:
    """
    Test case to check if latent history is returned correctly.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = model.random_tensor((1, 3, dim, dim))
    history_size = config.get("num_inference_steps") + 1 - config.get("start_step")

    images = model(
        image=image,
        prompt=config.get("prompt"),
        num_inference_steps=config.get("num_inference_steps"),
        start_step=config.get("start_step"),
        output_type="pt",
        return_latent_history=True,
    )

    assert type(images) is torch.Tensor
    assert images.shape == (1, history_size, 3, dim, dim)


def test_no_classifier_free_guidance(
    model: ImageToImageDiffusion, config: dict
) -> None:
    """
    Test case to check if the ImageToImageDiffusion is working correctly when classifier
    free guidance is disabled.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = model.random_tensor((1, 3, dim, dim))

    images = model(
        image=image,
        prompt=config.get("prompt"),
        num_inference_steps=config.get("num_inference_steps"),
        start_step=config.get("start_step"),
        output_type="np",
        guidance_scale=1.0,  # setting guidance_scale <= 1.0 effectively disables classifier free guidance
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


if __name__ == "__main__":
    pytest.main([__file__])
