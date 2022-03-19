# Changes in version 0.2

* `Tensor` and `Vector` now rely on Stdlib's `std::shared_ptr` for internal implementation. This makes memory management thread safe and somewhat more efficient on MSVC.

* Tensor, tensor views and sparse matrices dimensions are now stored in a `Dimensions` object, while `Indices` is just a synonym for a vector of `index` integers.

* `Dimensions` and `Indices` allow construction from initializer lists.

* `Tensor` can also be built from nested initializer lists. The interpretation of nested lists is based on C, so that a list of lists is a matrix, with the innermost lists representing the rows. For example:
```
RTensor vector = {1.0, 2.0, 3.0};
RTensor matrix = {{1.0, 2.0}, {3.0, 4.0}};
RTensor tensor = {{{1.0}, {2.0}}, {{3.0}, {4.0}}}
```

* Formerly, `Tensor::ones(n)` and `Tensor::zeros(n)` would return n-by-n matrices filled with ones and zeros, following Matlab. Now they return 1-D tensors, as Numpy.

* `Tensor::at(n)` only works for 1-D tensors, unlike before, where it allowed accessing all elements sequentially.

* Tensor views rely on a new iteration mechanism, built on top of `RangeIterator`, a class that, given a set of ranges and dimensions, produces a sequence of integers corresponding to the positions in a tensor corresponding to that view.

* A new constant `tensor::_` can be used to represent all elements in a range, as in `A(_, range(0))`.

* A new flag `TENSOR_DEBUG` determines whether the library is built in "debug" mode. In particular, this activates some costly assertions that verify the user's input.

* Assertions no longer rely on `<cassert>`. Instead, when assertions are activated (`TENSOR_DEBUG` is `ON`) and are not satisfied, the library raises an exception, which may change depending on the type of assertion. Generally, it will be a child of `std::logic_error`, such as `std::out_of_range` or `tensor::invalid_assertion`.

* `tic()` and `toc()` now use C++ high-precision `std::chrono` functions.

* Random number generation is now implemented using STL C++ generators.

* Sparse<> is presently an alias for CSRMatrix<> a new class that makes it clear which implementation we use for the matrices. This means also separating functions into different headers in a `tensor/sparse/` folder, but everything can be accessed from `tensor/sparse.h`.


## Incompatible changes

* When creating views, as in `A(range(0), range(0,2), range(0,0))`, there is now a guarantee that the library will remove dimensions of size 1 only if a single-argument range was used. In the example here, the output is a rank-2 tensor with 3 components: the first index is removed because `range(0)` is there, but the last dimension is not removed because a two-argument range was used `range(0,0)`, even if it only has size 1. This option may be toggled with the CMake flag `TENSOR_RANGE_SQUEEZE`.

* The `Vector` and `SimpleVector` classes now have `cbegin()` and `cend()` methods that replace the former `begin_const()` and `end_const()`.

* The previous static initializers `igen`, `rgen`, `cgen` and `gen<value_type>` no longer exist. They became obsolete with the introduction of `std::initializer_list` and pollute the space of constructors.

* To avoid ambiguities, all constructors of the form `Tensor(index i0, index i1 ...)` have been eliminated. Use `Tensor::empty(i0, i1, ...)` instead.

* Similarly, `Indices` and `Booleans` no longer have a scalar constructor indicating the reserved size. Use the static function `Indices::empty` instead.

* Member variables in class Sparse are now privated (the public declaration was a typo).
