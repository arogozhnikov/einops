"""
This is a fake script, it is not used.
Seems github does not count contributions unless you have a setup.py
"""

__author__ = "Alex Rogozhnikov"

from setuptools import setup

setup(
    name="einops",
    version="0.7.0",
    description="A new flavour of deep learning operations",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arogozhnikov/einops",
    author="Alex Rogozhnikov",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 ",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="deep learning, neural networks, tensor manipulation, machine learning, "
    "scientific computations, einops",
    install_requires=[
        # no run-time or installation-time dependencies
    ],
)
