from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="stablefused",
    version="0.1.9",
    description="StableFused is a toy library to experiment with Stable Diffusion inspired by 🤗 diffusers and various other sources!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aryan V S",
    author_email="contact.aryanvs+stablefused@gmail.com",
    url="https://github.com/a-r-r-o-w/stablefused/",
    python_requires=">=3.8.0",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "accelerate==0.21.0",
        "diffusers==0.19.3",
        "ftfy==6.1.1",
        "imageio==2.31.1",
        "imageio-ffmpeg==0.4.8",
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
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Steps to publish:
# 1. Update version in setup.py
# 2. python setup.py sdist bdist_wheel
# 3. Check if everything works with testpypi:
#    twine upload --repository testpypi dist/*
# 4. Upload to pypi:
#    twine upload dist/*
