from setuptools import find_packages, setup


setup(
    name="uma-inverse",
    version="0.1.0",
    description="PairMixer-based inverse folding model for ligand-conditioned sequence design.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "pytorch-lightning>=2.2",
        "numpy>=1.24",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "prody>=2.4",
        "biopython>=1.83",
    ],
)
