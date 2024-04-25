from setuptools import find_packages, setup

setup(
    name="jjuke",
    version="0.0.7.4",
    description="Framework and utilities for Deep Learning models with Pytorch by JJukE",
    author="JJukE",
    author_email="psj9156@gmail.com",
    url="https://github.com/JJukE/JJuk_E.git",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "pytorch3d",
        "scikit-image",
        "omegaconf",
        "easydict",
        "tqdm",
        "einops"
    ],
    keywords=["JJukE", "jjuke"]
)
