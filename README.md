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
conda config --add channels conda-forge
conda create -n cpp-dev cxx-compiler cmake openblas arpack fftw gtest
```

3. Build the library. Here `build-conda` is the name of the subdirectory where the library will be compiled and linked.
```cmd
cmake -S . -B build-conda -G "NMake Makefiles"
cmake --build build-conda
```

4. Optionally test
```cmd
ctest --test-dir build-conda/tests
```

If you want to speed up development and build time:

a. Install Ninja using `conda install ninja`
b. Instead of using `-G "NMake Makefiles"` use `-G "Ninja"`
c. Invoke `cmake --build -j N` where `N` is the number of parallel jobs for building Tensor.
d. Tell OpenBLAS the number of threads to use by defining the environment variable `OPENBLAS_NUM_THREADS`.

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
cd build-msvc\tests
ctest
```

## Unix platforms

1. Install required dependencies. On Debian you need
  a. Some version of the OpenBLAS library, such as `libopenblas64-openmp-dev` for parallelized operations.
  b. The optional libraries `libfftw3-3`, `libarpack2-dev`, and `libgtest-dev`
  c. Tools to build the library, including `cmake`, `g++` (or other c++ compiler) and `pkg-config`. This last one is needed because of a bug in the installation of OpenBLAS.

3. Clone the repository. We assume the repo will end up in `~/tensor`
```cmd
git clone https://github.com/juanjosegarciaripoll/tensor ~/tensor
```

3. Configure and build
```sh
cmake -B build-debian -S . -G "Unix Makefiles" -DTENSOR_ARPACK=ON -DTENSOR_FFTW=ON -DTENSOR_TEST=ON
cmake --build
```

4. Optionally test the library
```cmd
(cd build-debian/tests && ctest)
```

5. Install. We assume that you want to place the library the `/usr/local/lib` directory tree
```sh
cmake --install PREFIX=/usr/local/lib
```
