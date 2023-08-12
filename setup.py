from setuptools import setup, find_packages

setup(
    name="stablefused",
    version="0.1.1",
    description="StableFused",
    long_description="StableFused is a toy library to experiment with Stable Diffusion inspired by 🤗 diffusers and various other sources!",
    long_description_content_type="text/markdown",
    author="Aryan V S",
    author_email="contact.aryanvs+stablefused@gmail.com",
    python_requires=">=3.6.0",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "accelerate==0.21.0",
        "diffusers==0.19.3",
        "ftfy==6.1.1",
        "imageio==2.31.1",
        "torch==2.0.1",
        "transformers==4.31.0",
        "matplotlib==3.7.2",
        "numpy==1.25.2",
        "scipy==1.11.1",
    ],
    extras_require={
        "dev": [
            "black==23.7.0",
            "pytest==7.4.0",
            "twine>=4.0.2",
        ]
    },
)
