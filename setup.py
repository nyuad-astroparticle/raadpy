import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="raadpy",
    version="1.2.2",
    author="NYUAD Astroparticle Lab (Panos Oikonomou)",
    author_email="po524@nyu.edu",
    description="Python wrapper for data analysis of the RAAD detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nyuad-astroparticle/raadpy",
    project_urls={
        "Bug Tracker": "https://github.com/nyuad-astroparticle/raadpy/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'matplotlib>=3.5.1',
        'numpy>=1.22.3',
        'astropy>=5.0.2',
        'pandas>=1.4.1',
        'requests>=2.27.1',
        'tqdm>=4.64',
        'lxml>=4.8.0',
	    'plotly>=5.8.2'
    ],
)
