# StableFused

[![PyPI version](https://badge.fury.io/py/stablefused.svg)](https://badge.fury.io/py/stablefused)

StableFused is a toy library to experiment with Stable Diffusion inspired by 🤗 diffusers and various other sources!

## Installation

It is recommended to use a virtual environment. You can use [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) to create one.

```bash
python -m venv venv
```

For usage, install the package from PyPI.

```bash
pip install stablefused
```

For development, fork the repository, clone it and install the package in editable mode.

```bash
git clone https://github.com/<YOUR_USERNAME>/stablefused.git
cd stablefused
pip install -e ".[dev]"
```

## Usage

Checkout the [examples](./examples) folder for notebooks 🥰

## Contributing

Contributions are welcome! Note that this project is not a serious implementation for training/inference/fine-tuning diffusion models. It is a toy library. I am working on it for fun and experimentation purposes (and because I'm too stupid to modify large codebases and understand what's going on).

As I'm not an expert in this field, I will have probably made a lot of mistakes. If you find any, please open an issue or a PR. I'll be happy to learn from you!

## Acknowledgements

The following sources have been very helpful in helping me understand Stable Diffusion. I highly recommend you to check them out!

- [🤗 diffusers](https://github.com/huggingface/diffusers)
- [Karpathy's gist on latent walking](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355)
- [Nateraw's stable-diffusion-videos](https://github.com/nateraw/stable-diffusion-videos)
- [🤗 Annotated Diffusion Blog](https://huggingface.co/blog/annotated-diffusion)
- [Keras CV](https://github.com/keras-team/keras-cv)
- [Lillian Weng's Blogs](https://lilianweng.github.io/)
- [Emilio Dorigatti's Blogs](https://e-dorigatti.github.io/)
- [The AI Summer Diffusion Models Blog](https://theaisummer.com/diffusion-models/)

## Results

### Visualization of diffusion process

Refer to the notebooks for more details and enjoy the denoising process!

##### Text to Image

These results are generated using the [Text to Image](https://github.com/a-r-r-o-w/stablefused/blob/main/examples/text_to_image_diffusion.ipynb) notebook.

<div align="center">
  <video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/9528329d-ddc4-461e-9664-cbab3eb97123" controls loop autoplay>
    Your browser does not support the video tag.
  </video>
</div>

##### Image to Image

These results are generated using the [Image to Image](https://github.com/a-r-r-o-w/stablefused/blob/main/examples/image_to_image_diffusion.ipynb) notebook.

<table>
<thead>
  <tr>
    <th>Source Image</th>
    <th colspan="2">Denoising Diffusion Process<br></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://mspoweruser.com/best-stable-diffusion-prompts/#1_The_Renaissance_Astrounaut" target="_blank" rel="noopener noreferrer">The Renaissance Astronaut</a></td>
    <td>High quality and colorful photo of Robert J Oppenheimer, father of the atomic bomb, in a spacesuit, galaxy in the background, universe, octane render, realistic, 8k, bright colors</td>
    <td>Stylistic photorealisic photo of Margot Robbie, playing the role of astronaut, pretty, beautiful, high contrast, high quality, galaxies, intricate detail, colorful, 8k</td>
  </tr>
  <tr>
    <td><img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/cb5da9ed-80b3-4cd6-8874-ed3353967042" /></td>
    <td colspan="2"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/a0623f50-308b-40e1-a331-a7073c10281d" controls loop autoplay> Your browser does not support the video tag. </video></td>
  </tr>
</tbody>
</table>

<details>
  <summary>PS</summary>
  The results from Image to Image Diffusion don't seem very great from my experimentation. It might be some kind of bug in my implementation, which I'll have to look into later...
</details>

### Understanding the effect of Guidance Scale

Guidance scale is a value inspired by the paper [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). The explanation of how CFG works is out-of-scope here, but there are many online sources where you can read about it (linked below).

- [Guidance: a cheat for diffusion models](https://sander.ai/2022/05/26/guidance.html)
- [Diffusion Models, DDPMs, DDIMs and CFG](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869)
- [Classifier-Free Guidance Scale](https://mccormickml.com/2023/02/20/classifier-free-guidance-scale/)

In short, guidance scale is a value that controls the amount of "guidance" used in the diffusion process. That is, the higher the value, the more closely the diffusion process follows the prompt. A lower guidance scale allows the model to be more creative, and work slightly different from the exact prompt. After a certain threshold maximum value, the results start to get worse, blurry and noisy.

Guidance scale values, in practice, are usually in the range 6-15, and the default value of 7.5 is used in many inference implementations. However, manipulating it can lead to some very interesting results. It also only makes sense when it is set to 1.0 or higher, which is why many implementations use a minimum value of 1.0.

But... what happens when we set guidance scale to 0? Or negative? Let's find out!

When you use a negative value for the guidance scale, the model will try to generate images that are the opposite of what you specify in the prompt. For example, if you prompt the model to generate an image of an astronaut, and you use a negative guidance scale, the model will try to generate an image of everything but an astronaut. This can be a fun way to generate creative and unexpected images (sometimes NSFW or absolute horrendous stuff, if you are not using a safety-checker model - which is the case with StableFused).

##### Results

The original images produced are too large to display in high quality here. You can find them in my [Drive](https://drive.google.com/drive/folders/13eZsi7y1LZxUHlaxagGTPS6pLwzBysU6?usp=sharing). These images are compressed from ~30 MB to ~6 MB in order for GitHub to accept uploads.

<details>
  <summary>
    Effect of Guidance Scale on Different Prompts
  </summary>

  | Effect of Guidance Scale on Different Prompts |
  | --- |
  | Each row has first column set to the `guidance_scale` value. Each column has the same prompt. <br /> **Column 1:** _Artistic image, very detailed cute cat, cinematic lighting effect, cute, charming, fantasy art, digital painting, photorealistic_ <br /> **Column 2:** _A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k_ <br /> **Column 3:** _A grand city in the year 2100, atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render_ <br /> **Column 4:** _Starry Night, painting style of Vincent van Gogh, Oil paint on canvas, Landscape with a starry night sky, dreamy, peaceful_ |
  | <img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/56d5f47d-b2c3-4a5a-924a-84edc857ebf3" /> |
</details>

<details>
  <summary>
    Effect of Guidance Scale with increased number of inference steps
  </summary>

  | Effect of Guidance Scale with increased number of inference steps |
  | --- |
  | Each row has first column set to the `guidance_scale` value. Columns have number of inference steps set to 3, 6, 12, 20, 25. <br /> **Prompt:** _Photorealistic illustration of a mystical alien creature, magnificent, strong, atomic, tyrannic, predator, unforgiving, full-body image_ |
  | <img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/eed3882d-b4bb-49b7-98e6-35bbd7faf519" /> |
</details>

