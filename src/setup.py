from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

xsimd_include = os.environ.get('XSIMD_INCLUDE_PATH', '../third_party/xsimd/include')

# Check if OpenMP is enabled
use_openmp = os.environ.get('_LOFOPENMP', '0') == '1'

# Base compile args
compile_args = [
    '-std=c++20',
    '-O3',
    '-march=native',
    '-fPIC',
]

# Add OpenMP flag if enabled
if use_openmp:
    compile_args.append('-fopenmp')

# Base link args
link_args = []
if use_openmp:
    link_args.append('-fopenmp')

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='LoFloat',
    version='0.1.0',
    ext_modules=[
        CppExtension(
            name='LoFloat',
            sources=[
                'LoPy_bind.cpp',
            ],
            include_dirs=[
                xsimd_include,
                '.',
                os.path.join(script_dir, '../third_party/xsimd/include'),
            ],
            extra_compile_args={
                'cxx': compile_args
            },
            extra_link_args=link_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=2.0.0'],
)