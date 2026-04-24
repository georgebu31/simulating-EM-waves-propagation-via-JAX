from setuptools import setup, find_packages

setup(
    name="jaxoptics",
    version="0.1.0",
    description="JAX-based Angular Spectrum Method library for EM wave propagation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax",
        "jaxlib",
        "optax",
        "matplotlib",
    ],
)
