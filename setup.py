from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

xsimd_include = os.environ.get('XSIMD_INCLUDE_PATH', 'third_party/xsimd/include')

use_openmp = os.environ.get('_LOFOPENMP', '0') == '1'

compile_args = [
    '-std=c++20',
    '-O3',
    '-fPIC',
]

if use_openmp:
    compile_args.append('-fopenmp')

link_args = []
if use_openmp:
    link_args.append('-fopenmp')

script_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='LoFloat',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='LoFloat.LoFloat',
            sources=[
                os.path.join(script_dir, 'src/LoPy_bind.cpp'),  # absolute path, no ambiguity
            ],
            include_dirs=[
                xsimd_include,
                os.path.join(script_dir, 'src/'),
                os.path.join(script_dir, 'third_party/xsimd/include'),
            ],
            extra_compile_args={'cxx': compile_args},
            extra_link_args=link_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=2.0.0'],
)