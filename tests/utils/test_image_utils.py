import imageio
import numpy as np
import os
import pytest
import tempfile

from PIL import Image
from imageio.plugins.ffmpeg import FfmpegFormat
from stablefused.utils import image_grid, pil_to_gif, pil_to_video

np.random.seed(42)


@pytest.fixture
def image_config():
    return {
        "width": 32,
        "height": 32,
        "channels": 3,
    }


@pytest.fixture
def num_images():
    return 8


def random_image(width, height, channels):
    random_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(random_image)


@pytest.fixture
def random_images(image_config, num_images):
    image_list = []
    for _ in range(num_images):
        image_list.append(
            random_image(
                width=image_config.get("width"),
                height=image_config.get("height"),
                channels=image_config.get("channels"),
            )
        )
    return image_list


@pytest.fixture
def temporary_gif_file():
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file:
        temp_filename = temp_file.name
        yield temp_filename
    os.remove(temp_filename)


@pytest.fixture(params=[".mp4", ".avi", ".mkv", ".mov", ".wmv"])
def temporary_video_file(request):
    with tempfile.NamedTemporaryFile(suffix=request.param, delete=False) as temp_file:
        temp_filename = temp_file.name
        yield temp_filename
    os.remove(temp_filename)


def test_image_grid(random_images, num_images):
    """Test that image grid is created correctly."""

    rows = 2
    cols = num_images // 2
    grid_image = image_grid(random_images, rows, cols)

    expected_width = random_images[0].width * cols
    expected_height = random_images[0].height * rows

    assert grid_image.width == expected_width
    assert grid_image.height == expected_height
    assert len(random_images) == rows * cols
    assert isinstance(grid_image, Image.Image)


def test_pil_to_gif(random_images, temporary_gif_file):
    """Test that PIL images are converted to GIF correctly."""

    pil_to_gif(random_images, temporary_gif_file, fps=1)

    assert os.path.isfile(temporary_gif_file)
    with Image.open(temporary_gif_file) as saved_gif:
        assert isinstance(saved_gif, Image.Image)
        assert saved_gif.is_animated
        assert saved_gif.n_frames == len(random_images)


def test_pil_to_video(random_images, temporary_video_file):
    """Test that PIL images are converted to video correctly."""
    pil_to_video(random_images, temporary_video_file, fps=1)

    assert os.path.isfile(temporary_video_file)
    try:
        video: FfmpegFormat.Reader = imageio.get_reader(
            temporary_video_file, format="ffmpeg"
        )
        assert video.count_frames() == len(random_images)
    except Exception as e:
        pytest.fail(f"Failed to open video file: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
