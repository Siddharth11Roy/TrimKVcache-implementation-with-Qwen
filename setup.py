from setuptools import setup, find_packages

setup(
    name="trimkv",
    version="0.1.0",
    description="TrimKV: learned token retention for memory-bounded KV cache (arXiv:2512.03324)",
    packages=find_packages(exclude=("examples", "train", "tests")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3",
        "transformers>=4.53",
        "accelerate>=0.33",
        "datasets>=2.20",
        "einops>=0.8",
        "numpy>=1.26",
        "tqdm>=4.66",
    ],
)
