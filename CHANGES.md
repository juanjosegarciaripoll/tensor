Changes in version 0.2
======================

* `Tensor` and `Vector` now rely on Stdlib's `std::shared_ptr` for internal implementation. This makes memory management thread safe and somewhat more efficient on MSVC.

* Tensor, tensor views and sparse matrices dimensions are now stored in a `Dimensions` object, while `Indices` is just a synonym for a vector of `index` integers.

* `Dimensions` and `Indices` allow construction from initializer lists.