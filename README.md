LoFloat is a library for simulating varying precision and length floating point formats.

This project is funded by an STTR grant with University of California, Berkeley and Arith Inc.

## Building

The CUDA GEMM path depends on a forked CUTLASS (`SudhanvaKulkarni123/cutlass`) that threads LoFloat accumulation parameters through the GEMM stack. It is vendored as a git submodule under `third_party/cutlass`.

Fresh clone:

```bash
git clone --recursive git@github.com:SudhanvaKulkarni123/LoFloat.git
```

Existing clone:

```bash
git submodule update --init --recursive
```

Build PyTorch extension either via `./install.sh` or `USE_CUDA=1 pip install -e /path/to/LoFloat --no-build-isolation`).
