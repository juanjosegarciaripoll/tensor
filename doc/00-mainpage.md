# Tensor C++ library {#mainpage}

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/juanjosegarciaripoll/tensor)](https://github.com/juanjosegarciaripoll/tensor/releases/latest)
[![GitHub](https://img.shields.io/github/license/juanjosegarciaripoll/tensor)](https://github.com/juanjosegarciaripoll/tensor/blob/main/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/juanjosegarciaripoll/tensor)

Tensor is a tensor algebra library for doing numerics with arrays of real and complex numbers, typically floating point ones. The main motivation of the library is to provide arrays with more than one or two indices, and allow the kind of computations that are easily done with Matlab, [Numpy], Yorick, to name some software that inspired this library. 

Some of the conventions used in this library are heavily used by Matlab, because the software was developed well before Python and Numpy became standards for scientific computing, and also before Modern C++ (C++11, C++14 and so on) became a reality. The library is slowly evolving and growing to adapt to these new standards and conventions, also improving usability, portability and overall performance.

- [Installation](installation.html)
- [Usage](usage.html)
- [The tensor object](tensor.html)
- [Tensor operations](tensor_operations.html)
- [Sparse matrices](sparse.html)
- Linear algebra

[Numpy]: https://numpy.org/doc/stable/
[Yorick]: http://yorick.sourceforge.net/
