from setuptools import setup, find_packages

setup(
    name="cavity_qed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "dataclasses>=0.6",
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy", "pylint"],
        "arc": ["arc-alkali-rydberg-calculator"],
    },
    python_requires=">=3.8",
)
