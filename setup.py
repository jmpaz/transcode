from setuptools import setup, find_packages
import re


def get_requirements():
    with open("requirements.txt") as f:
        required = f.read().splitlines()
    return [req for req in required if re.match(r"^(?!git\+)[\w-]+", req)]


setup(
    name="transcode",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={"console_scripts": ["transcode = transcode.cli:cli"]},
    author="jmpaz",
    url="https://github.com/jmpaz/transcode",
    python_requires=">=3.6",
)
