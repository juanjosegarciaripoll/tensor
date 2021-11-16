# DESCRIPTION

Tensor is a C++ library for numerical arrays and tensor objects and operations with them, designed to allow Matlab-style programming. Some of the key features:

1. Fundamental data structures
  - Arrays of indices
  - Multidimensional arrays of double precision, real and complex numbers
  - Sparse matrices of real and complex numbers

2. Operations
  - Algebraic operations between tensors, matrices and numbers.
  - Contraction among tensors (a generalization of matrix multiplication).
  - Tensor reshaping, index permutation, enlarging and contracting.

3. Linear algebra
  - Solving systems of linear equations
  - Matrix exponentiation
  - Eigenvalue problems
  - Few eigenvalues of sparse problems using Arpack
  - Singular value decompositions

4. Fast Fourier Transforms

The Tensor library is discussed in the [libtensor Google Group](https://groups.google.com/forum/#!forum/libtensor). Go to this forum for help in configuring and installing the library.

Bug reports should be submitted using the GitHub Issues interface, although limited support could be provided via the group.

# Building

## Anaconda

You can build the library using Anaconda or Miniconda environments, using the pre-packaged libraries from Conda-Forge.

1. Ensure your system has a c++ compiler installed. On Windows the current Python pipeline depends in [Microsoft Visual Build Tools 2017](https://aka.ms/vs/15/release/vs_buildtools.exe), which you can download and install for free.

2. It is recommended to create a development environment to build the C++ packages. This is the description of the environment
```yaml
channels:
  - conda-forge
dependencies:
  - cxx-compiler
  - cmake
  - openblas
  - arpack
  - fftw
  - gtest
```
Alternatively, you can achieve the same with the command line interface
```cmd
conda config --add conda-forge
conda create -n cpp-dev cxx-compiler cmake openblas arpack fftw gtest
```

3. Build the library
```cmd
cmake -S . -B build-conda -G "Nmake Makefiles"
cmake --build build-conda
```

4. Optionally test
```cmd
cd build-conda\test
ctest
```

## Microsoft Visual Build Tools

Install the latest Visual Studio tools for C++. I tend to use the command line version, without the editor, which is available under the name Microsoft Visual Build Tools.

1. Install [vcpkg](https://vcpkg.io/en/index.html). We will assume that vcpkg is installed under `c:\dev\vcpkg`

2. From vcpkg, install a blas-compatible library. An easy solution is to use openblas.
```cmd
c:\dev\vcpkg\vcpkg install openblas
```
Alternatively, download the [Intel-MKL library](https://registrationcenter.intel.com/en/products/download/3178/) and install the `intel-mkl` compatibility layer.
```cmd
c:\dev\vcpkg\vcpkg install openblas
```

3. Clone the repository. We assume the repo will end up in `c:\dev\tensor`
```cmd
git clone https://github.com/juanjosegarciaripoll/tensor
```

4. From the command line, configure the library for building with Microsoft Visual C++ compilers.
```cmd
cd c:\dev\tensor
cmake -B build-msvc -S . -G "Nmake Makefiles" -DCMAKE_TOOLCHAIN_FILE=c:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build-msvc
```

4. Optionally run the tests
```cmd
cd build-msvc\test
ctest
```
