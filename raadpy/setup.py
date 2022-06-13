import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="raadpy-nyuad-po524",
    version="0.0.1",
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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
