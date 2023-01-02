#pragma once

#include <tensor/tensor/types.h>

namespace tensor {

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims)
    : data_(static_cast<size_t>(new_dims.total_size())), dims_(new_dims) {}

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims, const Tensor<elt_t> &other)
    : data_(other.data_), dims_(new_dims) {
  tensor_assert(dims_.total_size() == ssize());
}

// NOLINTNEXTLINE(*-explicit-constructor)
// cppcheck-suppress noExplicitConstructor
template <typename elt_t>
Tensor<elt_t>::Tensor(const vector_type &data)
    : data_(data), dims_{data_.size()} {}

// NOLINTNEXTLINE(*-explicit-constructor)
// cppcheck-suppress noExplicitConstructor
template <typename elt_t>
Tensor<elt_t>::Tensor(vector_type &&data)
    : data_(std::move(data)), dims_{data_.ssize()} {}

// NOLINTNEXTLINE(*-explicit-constructor)
// cppcheck-suppress noExplicitConstructor
template <typename elt_t>
Tensor<elt_t>::Tensor(const std::vector<elt_t> &data)
    : data_(data.size()), dims_{data_.ssize()} {
  std::copy(data.begin(), data.end(), begin());
}

}  // namespace tensor
