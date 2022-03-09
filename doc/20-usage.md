# Usage {#usage}

Tensor is a C++ library that must be called and used from other C++ code. Before using it, the library must be built and installed somewhere in your computer, as described in the [Installation](#installation) section.

Once this has been done, you must create your own C++ project with your code and link it to the library. As an entry level setup, I recommend downloading [Visual Studio Code](https://code.visualstudio.com/) and using the example program that is available at my [GitHub account](https://github.com/juanjosegarciaripoll).

Even if you are familiar with C++ or scientific programming with Python or Matlab, there may be some discrepancies between your expectations and how this library is to be used.

## Design choices

### Column vs row-major indices {#columnmajor}

There are [two main conventions](https://en.wikipedia.org/wiki/Row-_and_column-major_order) for storing matrices and tensors in the programming world. They differ on how elements are arranged in memory, and which index is the most rapidly varying one when running over consecutive elements. Take for instance the following matrix
$$
A = \begin{pmatrix} 1 & 2 \\\\ 3 & 4 \end{pmatrix}
$$
This is a matrix or tensor with two indices whose elements we can access as `A[i,j]`. In *row-major order*, a matrix' elements are organized such that `A[i,j]` and `A[i,j+1]` would be arranged contiguously in memory. In the previous example, the matrix would point to a region in memory where data would be stored as
```
A -> {1, 2, 3, 4}
```
In *column-major order* we follow the opposite convention, and `A[i,j]` is contiguous to `A[i+1,j]`. This results in a different memory arrangement
```
A -> {1, 3, 2, 4}
```

In this library, we chose the *column-major order* for all tensors.  This is the usual convention in Fortran and it simplifies interfacing  to BLAS/LAPACK libraries. However, C++ libraries and Python/Numpy use row major, because that is closer to the way list comprehension or array creation works. You have to keep this difference in mind, because the proximity of data also influences the speed of certain operations, as well as the output of reshape operations.

### Automatic memory management {#tensor_memory}

The library conforms to the RAII phylosophy in C++, by which the objects in the library manage their own resources and are exception safe. In the following example, we pass a tensor by reference and create some matices `A`, `B` and `C` within the function. These matrices require some memory to store the information they have. They also have some lifetimes, determined by the C++ standard. When each of these objects ceases to exist, it will take care of also releasing the memory it has claimed for the elements of their tensors.
```{.cc}
Tensor<double> my_function(const Tensor<double> &X) {
  Tensor<double> A = X * 3.0;
  Tensor<double> B = Tensor<double>::eye(2,2);
  ...
  if (do_something) {
      Tensor<double> C = Tensor<double>::random(10,10);
      ...
      // resources from matrix C are deleted
  }
  // resources from matrix B are deleted
  return A;
  // matrix A continues its lifetime outside this function
}
```

### Sharing memory {#tensor_sharing}

The memory management is done using a copy-on-write pointer mechanism, which allows two arrays to share the same data until one of them modifies the shared memory.
```{.cc}
  Tensor<double> a = Tensor<double>::eye(2,2);
  Tensor<double> b = a;  // a and b point to the same array
  a.at(0,1) = 1.0;       // a points now to a different memory
```
Since copying is cheap, it becomes feasible to have multiple references to the same array in structures such as vectors, lists, etc. It also makes the actual coding straightforward, since no bookkeeping has to be done

\note Copy-on-write is subject to change in future releases, where we may adopt Numpy's approach of sharing memory unless the user copies the tensor.

### Error handling

The library adopts the C++ Core Guidelines, implementing various contracts that verify user input and some of the generated data.

- In debug mode, the contracts are always enforced and, in case of error, the library signals an exception.

- In release mode or when TENSOR_DEBUG is OFF, some of those contracts are deactivated, and the remaining ones result in a call to `std::terminate`.

This choice is by design. In principle, when Tensor receives wrong inputs, for instance wrong indices into tensors or wrong shapes for a tensor view, this must be considered a logical error from which one may not recovered.
However, it is acknowledged that some projects may need from a kind of error recovery, to close resources used during a project or to log the error.

- In production coude, you may install your own `std::terminate_handler` to handle those error conditions.

- In debug mode or when testing, `std::terminate_handler` still works, unless you want to capture exceptions to detect and debug those conditions using `try/catch`.
