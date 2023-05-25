from setuptools import find_packages, setup

setup(
    name="jjuke",
    version="0.0.0.4",
    description="utilities for AI models with Pytorch by JJukE",
    author="JJukE",
    author_email="psj9156@gmail.com",
    url="https://github.com/JJukE/JJuk_E.git",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "scikit-image",
        "point_cloud_utils",
        "PyMCubes",
        "omegaconf",
        "easydict",
        "tqdm",
        "timm"
    ],
    keywords=["JJukE", "jjuke"],
    entry_points={"console_scripts": ["JJukE=jjuke.main:main"]},
)
