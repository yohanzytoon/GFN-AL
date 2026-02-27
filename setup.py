from setuptools import find_packages, setup

setup(
    name="gfn_active_learning_project",
    version="0.1.0",
    description="GFlowNets + Active Learning for sample-efficient Scrabble exploration",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.11.2",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "hydra-core>=1.3.2",
        "torch==2.5.1",
        "botorch>=0.10.0",
        "gpytorch>=1.11",
        "tqdm>=4.66.0",
    ],
)
