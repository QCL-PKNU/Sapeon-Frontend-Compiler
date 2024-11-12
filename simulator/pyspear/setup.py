import os
from setuptools import setup, find_packages

setup_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(setup_dir, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="pyspear",
    version="0.1",
    author="dudaji",
    packages=find_packages(),
    install_requires=requirements,
    author_email="contact@dudaji.com",
    description="Sapeon Simulator Pyspear Package",
    python_requires=">=3.8",
)
