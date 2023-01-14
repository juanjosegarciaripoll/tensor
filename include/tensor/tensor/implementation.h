#pragma once

#include <tensor/tensor/types.h>

namespace tensor {

template <typename elt_t>
Tensor<elt_t>::Tensor(Dimensions new_dims)
    : data_(make_shared_array<elt_t>(new_dims.total_size_t())),
      dims_(std::move(new_dims)) {}

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

#ifdef TENSOR_COPY_ON_WRITE
template <typename elt_t>
void Tensor<elt_t>::ensure_unique_ignore_data() {
  if tensor_unlikely (!data_.unique()) {
    data_ = make_shared_array<elt_t>(size());
  }
}

template <typename elt_t>
typename Tensor<elt_t>::iterator Tensor<elt_t>::begin() {
  return appropriate(data_, size());
}
#endif

template <typename elt_t>
void Tensor<elt_t>::randomize_not_shared(default_rng_t &rng) noexcept {
  std::generate(unsafe_begin_not_shared(), unsafe_end_not_shared(),
                [&]() -> elt_t { return rand_full<elt_t>(rng); });
}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::empty(Dimensions &&dimensions) {
  return Tensor<elt_t>(std::move(dimensions),
                       make_shared_array<elt_t>(dimensions.total_size_t()));
}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::empty(const Dimensions &dimensions) {
  return Tensor<elt_t>(Dimensions(dimensions),
                       make_shared_array<elt_t>(dimensions.total_size_t()));
}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::zeros(Dimensions dimensions) {
  auto output = empty(std::move(dimensions));
  std::fill(output.unsafe_begin_not_shared(), output.unsafe_end_not_shared(),
            number_zero<elt_t>());
  return output;
}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::ones(Dimensions dimensions) {
  auto output = empty(std::move(dimensions));
  std::fill(output.unsafe_begin_not_shared(), output.unsafe_end_not_shared(),
            number_one<elt_t>());
  return output;
}

template <typename elt_t>
Tensor<elt_t> Tensor<elt_t>::eye(index_t rows, index_t cols) {
  auto output = zeros(rows, cols);
  for (index i = 0; i < rows && i < cols; ++i) {
    output.at(i, i) = number_one<elt_t>();
  }
  return output;
}

template <typename elt_t>
Tensor<elt_t> &Tensor<elt_t>::fill_with(elt_t value) {
  ensure_unique_ignore_data();
  std::fill(unsafe_begin_not_shared(), unsafe_end_not_shared(), value);
  return *this;
}

template <typename elt_t>
Tensor<elt_t> &Tensor<elt_t>::fill_with_zeros() {
  return fill_with(number_zero<elt_t>());
}

template <typename elt_t>
Tensor<elt_t> &Tensor<elt_t>::randomize(default_rng_t &rng) {
  ensure_unique_ignore_data();
  randomize_not_shared(rng);
  return *this;
}

}  // namespace tensor
