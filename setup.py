from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="unified-xai",
    version="0.1.0",
    description="Unified XAI: An Explainable AI library for interpretable machine learning, Deep Learning and Artifical Intelligence.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatyamSingh8306/unified-xai",
    author="Satyam Singh",
    author_email="satyamsingh7734@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "lime>=0.2.0",
        "shap>=0.40.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "pillow>=8.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "dashboard": ["streamlit>=1.0.0", "dash>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
)
