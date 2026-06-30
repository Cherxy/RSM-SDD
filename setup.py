from setuptools import find_packages, setup


setup(
    name="pama_sdd",
    version="0.1.0",
    description="PAMA-SDD reproduction on top of a lightweight MDistiller-style framework.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "PyYAML>=6.0",
        "tqdm>=4.66.0",
    ],
)

