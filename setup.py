from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ProbeLens",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A barebones linear probe toolkit for TransformerLens and SAELens.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ProbeLens",
    packages=find_packages(),
    install_requires=[
        "transformer_lens",
        "sae-lens",
        "scikit-learn",
        "plotly",
        "safetensors",
        "wandb",
        "torch",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)