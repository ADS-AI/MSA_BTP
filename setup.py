import codecs
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.10'
DESCRIPTION = 'MSA Toolbox'
LONG_DESCRIPTION = 'Model Stealing attack'

install_requires = [
    "numpy>=1.18.0",
    "scikit-learn>=0.22.2,<1.2.0",
    "six",
    "setuptools",
    "tqdm",
    "scipy>=1.7.3",
    "matplotlib>=3.5.1"
]


# Setting up
setup(
    name="msa_toolbox",
    version=VERSION,
    author="Akshit, Sumit, Khushdev",
    author_email="sumit20249@iiitd.ac.in",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=install_requires,
    keywords=['msa'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    extras_require={
        "pytorch": ["torch", "torchvision"],
        "pytorch_image": ["torch", "torchvision", "kornia", "Pillow", "ffmpeg-python", "opencv-python"],
        "pytorch_audio": ["torch", "torchvision", "torchaudio", "pydub", "resampy", "librosa"],
        "xgboost": ["xgboost"],
        "all": [
            "torch",
            "torchvision",
            "xgboost",
            "pandas",
            "kornia",
            "matplotlib",
            "Pillow",
            "statsmodels",
            "pydub",
            "resampy",
            "ffmpeg-python",
            "cma",
            "librosa",
            "opencv-python",
            "numba",
        ],
        "non_framework": [
            "matplotlib",
            "Pillow",
            "statsmodels",
            "pydub",
            "resampy",
            "ffmpeg-python",
            "cma",
            "pandas",
            "librosa",
            "opencv-python",
            "pytest",
            "pytest-flake8",
            "pytest-mock",
            "pytest-cov",
            "codecov",
            "requests",
            "sortedcontainers",
            "numba",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
)
