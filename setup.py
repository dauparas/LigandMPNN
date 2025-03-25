from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()
    
setup(
    name="ligandmpnn",
    version="0.1.2",
    description="a pip installable version of LigandMPNN with pre-trained models included",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dauparas/LigandMPNN",
    author="originally and primarily authored by Justas Dauparas. installable version by markus.",
    author_email = "justas@uw.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    install_requires=["torch==2.2.1",
                      "biopython==1.79", 
                      "filelock==3.13.1", 
                      "fsspec==2024.3.1", 
                      "Jinja2==3.1.3", 
                      "MarkupSafe==2.1.5", 
                      "mpmath==1.3.0", 
                      "networkx==3.2.1", 
                      "numpy==1.23.5", 
                      "ProDy==2.4.1", 
                      "pyparsing==3.1.1", 
                      "scipy==1.12.0", 
                      "sympy==1.12", 
                      "typing_extensions==4.10.0", 
                      "ml-collections==0.1.1", 
                      "dm-tree==0.1.8", 
                      "setuptools"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2", "pytest-cov>=4.0", "wheel"],
    },
    python_requires=">=3.11, <3.12", 
    entry_points={
       'console_scripts': [
            'ligandmpnn = ligandmpnn.run:main',
            'mpnn = ligandmpnn.run:main',
            'ligand_mpnn = ligandmpnn.run:main',
        ],
    },
    include_package_data=True,
    package_data={'ligandmpnn': ['inputs/*',
                                 'data/model_params/*',
                                 'scripts/*']},
    
)
