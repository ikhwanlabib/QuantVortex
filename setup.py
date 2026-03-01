from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantvortex",
    version="1.0.0",
    author="QuantVortex Team",
    author_email="quantvortex@example.com",
    description="A production-grade quantitative trading engine combining statistical arbitrage, momentum, mean reversion, ML alpha, and reinforcement learning strategies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuantProject1/QuantVortex",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantvortex=quantvortex.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "quantvortex": ["config/*.yaml"],
    },
)
