from setuptools import setup, find_packages


# Dynamically determine which packages to include
def find_default_packages():
    return find_packages(where="src", exclude=["javad.extras"])


# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup configuration
setup(
    name="javad",
    version="0.1.0",
    description="JaVAD: Just Another Voice Activity Detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sergey Skrebnev",
    author_email="sergey.skrebnev@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    packages=find_default_packages(),  # Default to core packages only
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "javad.assets": ["*.pt", "mel_filters.npz"],
    },
    extras_require={"extras": ["soundfile>=0.12.0"]},
)
