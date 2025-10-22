"""
Setup script para o projeto Cooperative Demand Response with MARL
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cooperative-demand-response-marl",
    version="0.1.0",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="Sistema de Resposta Cooperativa à Demanda com Aprendizado por Reforço Multi-Agente",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/cooperative-demand-response-marl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cdrm-train=scripts.train:main",
            "cdrm-evaluate=scripts.evaluate:main",
        ],
    },
)