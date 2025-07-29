from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

#run file with python3 setup.py build_ext --inplace
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def get_torch_include_paths():
    """Get the required PyTorch include directories."""
    import torch
    torch_path = torch.__path__[0]
    print(torch_path)
    return [
        os.path.join(torch_path, "include"),
        os.path.join(torch_path, "include", "torch", "csrc", "api", "include"),
    ]


ext_modules = [
    Extension(
        "LoFloat",  # name of the resulting Python module
        ["Torch_overload.cpp"],  # path to your C++ source(s)
        include_dirs=[
            # Path to pybind11 headers
            str(get_pybind_include()),
            *get_torch_include_paths(),
        ],
        language="c++",
        # Add -mmacosx-version-min=10.12 for macOS target
        extra_compile_args=["-O3", "-std=c++20", "-w"],
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