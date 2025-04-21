# run pip install -e .

from setuptools import setup, find_packages

setup(
    name="privacy_learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'openai',
        'numpy',
        'pandas'
    ],
)