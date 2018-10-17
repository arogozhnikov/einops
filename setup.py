__author__ = 'Alex Rogozhnikov'

from setuptools import setup

setup(
    name="einops",
    version='0.1.0',
    description="A new flavour of deep learning operations",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',

    url='https://github.com/arogozhnikov/einops',

    # Author details
    author='Alex Rogozhnikov',

    packages=['einops', 'einops.layers'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],

    # What does your project relate to?
    keywords='deep learning, neural networks, tensor manipulation',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'numpy',
    ],
)
