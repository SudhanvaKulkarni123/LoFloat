from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        "LoFloat",  # name of the resulting Python module
        ["pybind_instantiations.cpp"],  # path to your C++ source(s)
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++20"],
    ),
]

setup(
    name="LoFloat",
    version="0.0.1",
    author="Sudhanva Kulkarni",
    author_email="sudhanvakulkarni@berkeley.edu",
    description="A module to simulate custom float and integer formats",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["pybind11>=2.6"],
)
