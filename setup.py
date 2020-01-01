import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyGOURGS",
    version="0.0.14",
    author="Sohrab Towfighi",
    author_email="sohrab.towfighi@mail.utoronto.ca",
    description="Global optimization by uniform random global search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pySRURGS/pyGOURGS",
    packages=["pyGOURGS"],
    py_modules=["pyGOURGS"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)