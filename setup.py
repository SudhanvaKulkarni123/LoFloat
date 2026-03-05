from setuptools import setup, find_packages
import os
import subprocess
import torch

TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), 'lib')

# --- CUDA home detection ---
use_cuda = os.environ.get('USE_CUDA', '0') == '1'
if use_cuda and not os.environ.get('CUDA_HOME'):
    nvcc = subprocess.run(['which', 'nvcc'], capture_output=True, text=True).stdout.strip()
    if nvcc:
        print("found nvcc at:", nvcc)
        os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvcc))
    else:
        print("nvcc not found in PATH. Checking common CUDA installation directories...")
        candidates = ['/usr/local/cuda', '/usr/cuda', '/opt/cuda']
        for path in candidates:
            if os.path.isdir(path):
                os.environ['CUDA_HOME'] = path
                break
        else:
            raise EnvironmentError("Could not find CUDA. Please set CUDA_HOME manually.")

# --- Extension ---
if use_cuda:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    cxx_args = ['-std=c++20', '-O3', '-fPIC', '-DUSE_CUDA']
    nvcc_args = ['-std=c++20', '-O3', '-DUSE_CUDA']
    use_openmp = os.environ.get('_LOFOPENMP', '0') == '1'
    link_args = ['-fopenmp'] if use_openmp else []
    if use_openmp:
        cxx_args.append('-fopenmp')
    # --- Paths and flags ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs = [
        os.path.join(script_dir, 'src/'),
        os.path.join(script_dir, 'third_party/xsimd/include'),
    ]
    ext = CUDAExtension(
        name='LoFloat.LoFloat',
        sources=[
            os.path.join(script_dir, 'src/LoPy_bind.cpp'),
            os.path.join(script_dir, 'src/Lof_kernel.cu'),
        ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
        extra_link_args=link_args,
        runtime_library_dirs=[TORCH_LIB],
    )
else:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    # --- Paths and flags ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xsimd_include = os.environ.get('XSIMD_INCLUDE_PATH', 'third_party/xsimd/include')
    use_openmp = os.environ.get('_LOFOPENMP', '0') == '1'
    link_args = ['-fopenmp'] if use_openmp else []
    include_dirs = [
        xsimd_include,
        os.path.join(script_dir, 'src/'),
        os.path.join(script_dir, 'third_party/xsimd/include'),
    ]
    cxx_args = ['-std=c++20', '-O3', '-fPIC']
    if use_openmp:
        cxx_args.append('-fopenmp')
    ext = CppExtension(
        name='LoFloat.LoFloat',
        sources=[
            os.path.join(script_dir, 'src/LoPy_bind.cpp'),
        ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': cxx_args},
        extra_link_args=link_args,
        runtime_library_dirs=[TORCH_LIB],
    )

# --- Setup ---
setup(
    name='LoFloat',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)