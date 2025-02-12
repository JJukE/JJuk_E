from setuptools import find_packages, setup

setup(
    name="jjuke",
    version="1.0.0",
    description="Framework for training Deep Learning networks by JJukE",
    author="JJukE",
    author_email="psj9156@gmail.com",
    url="https://github.com/JJukE/JJuk_E.git",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "accelerate",
        "transformers",
        "diffusers",
        "omegaconf",
        "easydict",
        "tqdm",
        "wandb",
        "rich",
    ],
    keywords=["JJukE", "jjuke"]
)
