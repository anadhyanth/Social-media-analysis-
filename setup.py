from setuptools import setup, find_packages

setup(
    name="social_media_analysis",
    version="1.0.0",
    author="B. Anadhyanth",
    description="Machine Learning based Social Media Analysis and Sentiment Classification System",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "nltk",
        "wordcloud",
        "joblib"
    ],
    python_requires=">=3.8",
)