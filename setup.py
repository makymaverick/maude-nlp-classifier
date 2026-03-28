from setuptools import setup, find_packages

setup(
    name="maude-nlp-classifier",
    version="0.1.0",
    description="NLP severity classifier for MAUDE medical device adverse event reports",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "scikit-learn>=1.4.0",
        "joblib>=1.3.2",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "requests>=2.31.0",
        "streamlit>=1.34.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
)
