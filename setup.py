from setuptools import find_packages, setup

setup(
    name="jjuke",
    version="0.0.3.2",
    description="Modules and utilities for Deep Learning training or inference with Pytorch by JJukE",
    author="JJukE",
    author_email="psj9156@gmail.com",
    url="https://github.com/JJukE/JJuk_E.git",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "pytorch3d",
        "trimesh",
        # "open3d",
        # "open3d-python==0.7.0.0",
        "scikit-image",
        "point_cloud_utils",
        "PyMCubes",
        "omegaconf",
        "easydict",
        "tqdm",
        "timm",
        "einops==0.6.1"
    ],
    keywords=["JJukE", "jjuke"],
    entry_points={"console_scripts": ["JJukE=jjuke.main:main"]},
)
