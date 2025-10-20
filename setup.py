from setuptools import setup, find_packages

setup(
    name="flora",
    version="0.1.0",
    description="Fractional LoRA (FLoRA) - Token-efficient fine-tuning via sketch-based token selection",
    author="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "bitsandbytes>=0.43.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        # "flash-attn>=2.4.0",
    ],
)
