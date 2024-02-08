from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="repro_gan",
    version="0.0.1",
    description="PyTorch 1D GAN library that provides you with modules, utilities, and metrics to create GAN models easily",
    package_dir={"": "repro_gan"},
    packages=find_packages(where="repro_gan"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshparksj/repro-gan",
    author="Joshua Park",
    author_email="joshparksj@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.23.1",
        "scipy>=1.9.0",
        "torch>=1.12.1",
        "matplotlib>=3.5.2",
        "scikit-learn>=1.1.2",
        "tensorboard>=2.7.0"
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)