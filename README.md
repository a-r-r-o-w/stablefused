# StableFused

<h4 align="center">
  
  <a href="https://pypi.org/project/stablefused/">
    <img src="https://img.shields.io/pypi/v/stablefused" alt="pypi">
  </a>

  <a href="https://a-r-r-o-w.github.io/stablefused/">
    <img src="https://img.shields.io/badge/stablefused-docs-blue" alt="docs">
  </a>

</h4>

StableFused is a toy library to experiment with Stable Diffusion inspired by 🤗 diffusers and various other sources! One of the main reasons I'm working on this project is to learn more about Stable Diffusion, and generative models in general. It is my current area of research at university. 

## Installation

It is recommended to use a virtual environment. You can use [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) to create one.

Unix:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```PowerShell
python -m venv venv
venv\Scripts\activate
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

Checkout the [examples](https://github.com/a-r-r-o-w/stablefused/tree/main/examples) folder for notebooks 🥰

## Contributing

Contributions are welcome! Note that this project is not a serious implementation for training/inference/fine-tuning diffusion models. It is a toy library. I am working on it for fun and experimentation purposes (and because I'm too stupid to modify large codebases and understand what's going on).

As I'm not an expert in this field, I will have probably made a lot of mistakes. If you find any, please open an issue or a PR. I'll be happy to learn from you!

## Acknowledgements/Resources

The following sources have been very helpful to me in understanding Stable Diffusion. I highly recommend you to check them out!

- [🤗 diffusers](https://github.com/huggingface/diffusers)
- [Karpathy's gist on latent walking](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355)
- [Nateraw's stable-diffusion-videos](https://github.com/nateraw/stable-diffusion-videos)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff/)
- [RunwayML StableDiffusion](https://github.com/runwayml/stable-diffusion)
- [🤗 Annotated Diffusion Blog](https://huggingface.co/blog/annotated-diffusion)
- [Keras CV](https://github.com/keras-team/keras-cv)
- [Lillian Weng's Blogs](https://lilianweng.github.io/)
- [Emilio Dorigatti's Blogs](https://e-dorigatti.github.io/)
- [The AI Summer Diffusion Models Blog](https://theaisummer.com/diffusion-models/)

## Results

### Visualization of diffusion process

Refer to the notebooks for more details and enjoy the denoising process!

<details>
  <summary> Text to Image </summary>

  These results are generated using the [Text to Image](https://github.com/a-r-r-o-w/stablefused/blob/main/examples/text_to_image_diffusion.ipynb) notebook.

  <div align="center">
    <video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/dd087a92-7111-4880-b207-0e309ce9aa87" controls loop autoplay>
      Your browser does not support the video tag.
    </video>
  </div>
</details>

<details>
  <summary> Image to Image </summary>

  These results are generated using the [Image to Image](https://github.com/a-r-r-o-w/stablefused/blob/main/examples/image_to_image_diffusion.ipynb) notebook.

  <table>
  <thead>
    <tr>
      <th>Source Image</th>
      <th colspan="2">Denoising Diffusion Process</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://mspoweruser.com/best-stable-diffusion-prompts/#1_The_Renaissance_Astrounaut" target="_blank" rel="noopener noreferrer">The Renaissance Astronaut</a></td>
      <td><i>High quality and colorful photo of Robert J Oppenheimer, father of the atomic bomb, in a spacesuit, galaxy in the background, universe, octane render, realistic, 8k, bright colors</i></td>
      <td><i>Stylistic photorealisic photo of Margot Robbie, playing the role of astronaut, pretty, beautiful, high contrast, high quality, galaxies, intricate detail, colorful, 8k</i></td>
    </tr>
    <tr>
      <td><img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/03bd14f5-f0d1-460b-8b5b-ad9f0a7f79f5" /></td>
      <td colspan="2"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/285dc6d7-883c-47a2-a89c-2253ded98340" controls loop autoplay> Your browser does not support the video tag. </video></td>
    </tr>
  </tbody>
  </table>

  <details>
    <summary> PS </summary>
    The results from Image to Image Diffusion don't seem very great from my experimentation. It might be some kind of bug in my implementation, which I'll have to look into later...
  </details>
</details>

### Text to Video

There is a lot of ongoing research on the generation of videos from text prompts. It is also my current area of research at university. The implementation here is adapted from [AnimateDiff](https://animatediff.github.io/).

There is immense potential in developing this kind of technology and its possible usecases are unlimited - personalized educational content, marketing and advertising, creativity and art, etc. to name a few. Imagine a world where you have your own personal ChatGPT/Bard like assistants for visual learning - a model that can generate [3Blue1Brown](https://www.youtube.com/c/3blue1brown) style videos explaining science topics, or depict a story! Current models are not that capable yet, but this is where we are headed, I think, and is what me and my team are researching on. The future of this technology will be fascinating to witness!

<details>
  <summary> Text to Video </summary>

  These results are generated using the [Text to Video](https://github.com/a-r-r-o-w/stablefused/blob/main/examples/text_to_video_diffusion.ipynb) notebook.

  <div align="center">
  
  | Text to Video |
  | :-: |
  | _An astronaut floating in space, interstellar, black background with stars, photorealistic, high quality, 8k_ |
  | <video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/72a10ec3-ccb2-45db-8e49-8752e705c57d" controls loop autoplay> Your browser does not support the video tag. </video> |
  | _A mighty pirate ship sailing through the sea, unpleasant, thundering roar, dark night, starry night, high quality, photorealistic, 8k_ |
  | <video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/b883e6e3-80b6-4b53-b91c-60edd715941c" controls loop autoplay> Your browser does not support the video tag. </video> |

  </div>
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
  | Each image is sampled with the same prompt and seed to ensure only the guidance scale plays a role. <br /> **Column 1:** _Artistic image, very detailed cute cat, cinematic lighting effect, cute, charming, fantasy art, digital painting, photorealistic_ <br /> **Column 2:** _A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k_ <br /> **Column 3:** _A grand city in the year 2100, atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render_ <br /> **Column 4:** _Starry Night, painting style of Vincent van Gogh, Oil paint on canvas, Landscape with a starry night sky, dreamy, peaceful_ |
  | <div align="center"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/546ca306-4e3b-42e3-9816-40ba9a2b060d" controls loop autoplay> Your browser does not support the video tag. </video></div> |
</details>

<details>
  <summary>
    Effect of Guidance Scale with increased number of inference steps
  </summary>

  | Effect of Guidance Scale with increased number of inference steps |
  | --- |
  | Columns have number of inference steps set to 3, 6, 12, 20, 25. <br /> **Prompt:** _Photorealistic illustration of a mystical alien creature, magnificent, strong, atomic, tyrannic, predator, unforgiving, full-body image_ |
  | <div align="center"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/55609d17-df70-4cf8-a19b-a3cd87be912a" controls loop autoplay> Your browser does not support the video tag. </video></div> |
  | <div align="center"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/7a6c507c-2694-433b-a695-5afa4f380718" controls loop autoplay> Your browser does not support the video tag. </video></div> |
</details>

### Latent Walk

Generative models, like the ones used in Stable Diffusion, learn a latent representation of the world. A latent representation is a low-dimensional vector space embedding of the world. In the case of SD, this latent representation is learnt by training on text-image pairs. This representation is used to generate samples given a prompt and a random noise vector. The model tries to predict and remove noise from the random noise vector, while also aligning the vector to the prompt. This results in some interesting properties of the latent space.

Stable Diffusion models (atleast, the models used here) learn two latent representations - one of the NLP space for prompts, and one of the image space. These latent representations are continuous. If we choose two vectors in the latent space to sample from, we get two different/similar images depending on how different the chosen vectors are. This is the basis of latent walking. We can choose two vectors in the latent space, and sample from the latent path between them. This results in a smooth transition between the two images.

<details>
  <summary> Similar Image Generation by sampling latent space </summary>
  
  The results below show just how information rich the latent space of these stable diffusion models are.
  
  <table>
  <thead>
    <tr>
      <th>Source Image</th>
      <th>Latent Walks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">
        <i> Large futuristic mechanical robot in the foreground of a baroque-style battle scene, photorealistic, high quality, 8k </i>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/1dcbf894-4988-45a5-b21f-8ffa4c3fb40e">
      </td>
      <td>
        <img src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/07c4354f-1580-4c60-9e46-fbc539dbc716">
      </td>
    </tr>
  </tbody>
  </table>
</details>

<details>
  <summary>
    Generating Latent Walk videos
  </summary>

  | Generating Latent Walk videos |
  | --- |
  | **Prompt 1:** _A dog chasing a cat in a thrilling backyard scene, high quality and photorealistic_  <br /> **Prompt 2:** _A determined dog in hot pursuit, with stunning realism, octane render_  <br /> **Prompt 3:** _A thrilling chase, dog behind the cat, octane render, exceptional realism and quality_  <br /> **Prompt 4:** _The exciting moment of a cat outmaneuvering a chasing dog, high-quality and photorealistic detail_  <br /> **Prompt 5:** _A clever cat escaping a determined dog and soaring into space, rendered with octane render for stunning realism_  <br /> **Prompt 6:** _The cat's escape into the cosmos, leaving the dog behind in a scene,high quality and photorealistic style_  <br /> |
  | <div align="center"><video src="https://github.com/a-r-r-o-w/stablefused/assets/72266394/5caf4bc3-d615-41d4-959a-38921e9e4516" controls loop autoplay> Your browser does not support the video tag. </video></div> |

  Note that these results aren't very good. I tried different seeds but for this story, I couldn't make a great video. I did try some other prompts and got better results, but I like this story so I'm sticking with it 🤓
  You can improve the results by using better prompts and increasing the number of interpolation and inference steps.
</details>

## Future

At the moment, I'm not sure if I'll continue to expand on this project, but if I do, here are some things I have in mind (in no particular order, and for documentation purposes):

- Add support for more techniques of inference - explore new sampling techniques and optimize diffusion paths
- Implement and stay up-to-date with the latest papers in the field
- Removing 🧨 diffusers as a dependency by implementing all required components myself
- Create user-friendly web demos or GUI tools to make experimentation easier.
- Add LoRA, training and fine-tuning support
- Improve codebase, documentation and tests
- Improve support for not only Stable Diffusion, but other diffusion techniques, involving but not limited to audio, video, etc.

## License

MIT
