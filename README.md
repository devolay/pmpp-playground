# PMPP Playground (C & CUDA)
    
A repository for experimenting while reading **Programming Massively Parallel Processors** book.

## Structure

```text
pmpp-playground/
├── CMakeLists.txt           # Top-level build
├── common/                  # Reusable library (C + header-only CUDA helpers)
└── chapters/                # Chapter-specific code
```

## Build

Requirements:
- CMake >= 3.22
- CUDA toolkit installed (for CUDA kernels)
- A C compiler (gcc/clang/MSVC)

```bash
cd pmpp-playground
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Notes

- `common/timer.h` offers a simple wall-clock timer in C.
- `common/cuda_check.h` provides `CUDA_CHECK` for ergonomic CUDA error handling.
- `common/cuda_utils.cuh` has small, header-only CUDA helpers (grid dims, etc.).
```