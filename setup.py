from setuptools import setup, find_packages

setup(
    name="stablefused",
    version="0.1.0",
    description="StableFused",
    author="Aryan V S",
    author_email="contact.aryanvs+stablefused@gmail.com",
    python_requires=">=3.6.0",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "diffusers==0.19.3",
        "torch==2.0.1",
        "transformers==4.31.0",
        "scipy",
        "ftfy",
        "ipython",
        "matplotlib",
        "pytest>=7.4.0",
    ],
)
