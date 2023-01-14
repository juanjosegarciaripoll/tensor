#pragma once

#include <tensor/tensor/types.h>

namespace tensor {

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims)
    : data_(make_shared_array<elt_t>(new_dims.total_size_t())),
      dims_(new_dims) {}

template <typename elt_t>
Tensor<elt_t>::Tensor(const Dimensions &new_dims, const Tensor<elt_t> &other)
    : data_(other.data_), dims_(new_dims) {
  tensor_assert(dims_.total_size() == ssize());
}

template <typename elt_t>
Tensor<elt_t>::Tensor(Dimensions dimensions, shared_array<elt_t> data) noexcept
    : data_(std::move(data)), dims_(std::move(dimensions)) {}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::from_pointer(Dimensions dimensions, elt_t *data) {
  return Tensor<elt_t>(std::move(dimensions),
                       make_shared_array_from_ptr<elt_t>(data));
}

// NOLINTNEXTLINE(*-explicit-constructor)
// cppcheck-suppress noExplicitConstructor
template <typename elt_t>
Tensor<elt_t>::Tensor(const std::vector<elt_t> &data)
    : data_(make_shared_array<elt_t>(data.size())),
      dims_{static_cast<index_t>(data.size())} {
  std::copy(data.begin(), data.end(), unsafe_begin_not_shared());
}

}  // namespace tensor
