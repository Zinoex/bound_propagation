import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bound_propagation",
    version="0.3.2",
    author="Frederik Baymler Mathiesen",
    author_email="frederik@baymler.com",
    description="Bound propagation in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zinoex/bound_propagation",
    project_urls={
        "Bug Tracker": "https://github.com/zinoex/bound_propagation/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'torch',
        'numpy'
    ]
)
