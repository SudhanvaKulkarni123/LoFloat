#pragma once

// Compiler detection
#if defined(_MSC_VER)
  #define LOFLOAT_COMPILER_MSVC 1
#else
  #define LOFLOAT_COMPILER_MSVC 0
#endif

// CUDA
#if defined(__CUDACC__)
  #define LOFLOAT_CUDA 1
#else
  #define LOFLOAT_CUDA 0
#endif

// Attributes
#if LOFLOAT_CUDA
  #define LOFLOAT_HOST __host__
  #define LOFLOAT_DEVICE __device__
  #define LOFLOAT_HOST_DEVICE __host__ __device__
  #define LOFLOAT_GLOBAL __global__
#else
  #define LOFLOAT_HOST
  #define LOFLOAT_DEVICE
  #define LOFLOAT_HOST_DEVICE
  #define LOFLOAT_GLOBAL
#endif

// Inline control
#if LOFLOAT_CUDA
  #define LOFLOAT_FORCEINLINE __forceinline__
  #define LOFLOAT_NOINLINE __noinline__
  #define LOFLOAT_INLINE __inline__
#elif LOFLOAT_COMPILER_MSVC
  #define LOFLOAT_FORCEINLINE __forceinline
  #define LOFLOAT_NOINLINE __declspec(noinline)
  #define LOFLOAT_INLINE __inline
#else
  #define LOFLOAT_FORCEINLINE inline __attribute__((always_inline))
  #define LOFLOAT_NOINLINE __attribute__((noinline))
    #define LOFLOAT_INLINE inline
#endif

//Restrict
#if LOFLOAT_COMPILER_MSVC
  #define LOFLOAT_RESTRICT __restrict
#else
  #define LOFLOAT_RESTRICT __restrict__
#endif
