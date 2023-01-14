#pragma once
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#ifndef TENSOR_TENSOR_BASE_H
#define TENSOR_TENSOR_BASE_H

/** Flag defining the order of elements in the arrays. */
#define TENSOR_COLUMN_MAJOR_ORDER 1

#include <vector>
#include <array>
#include <tensor/exceptions.h>
#include <tensor/numbers.h>
#include <tensor/vector.h>
#include <tensor/indices.h>
#include <tensor/detail/initializer.h>
#include <tensor/detail/shared_ptr.h>
#include <tensor/rand.h>
#include <tensor/ranges.h>
#include <tensor/tensor/iterator.h>

/*!\addtogroup Tensors*/
/* @{ */
namespace tensor {

template <typename elt_t>
class TensorView;
template <typename elt_t>
class MutableTensorView;

/**An N-dimensional array of numbers. A Tensor is a multidimensional array of
   numbers. Their behavior is similar to Matlab's arrays in that they can store
   only numbers, be accessed with one or more indices using the () or []
   syntaxes, reshaped, sliced, and all that with an automated memory management.
   \see \ref tensor
*/
template <typename elt>
class Tensor {
 public:
  /** The type of this tensor's elements */
  using elt_t = elt;
  /** The type of this tensor's elements */
  using value_type = elt;
  /** Random access iterator type */
  using iterator = elt_t *;
  /** Random access iterator type to const */
  using const_iterator = const elt_t *;

  /**Constructs an empty Tensor.*/
  Tensor() = default;

  ~Tensor() = default;

  /**Constructs a 1-D Tensor from a vector.*/
  Tensor(const std::vector<elt_t> &data);

  /**Optimized copy constructor.*/
  Tensor(const Tensor &other) = default;

  /**Optimized move constructor. */
  Tensor(Tensor &&other) noexcept = default;

  /**Implicit coercion and transformation to a different type. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  template <typename e2>
  // cppcheck-suppress noExplicitConstructor
  Tensor(const Tensor<e2> &other)
      : data_(make_shared_array<elt>(other.size())), dims_(other.dimensions()) {
    std::copy(other.begin(), other.end(), begin());
  }

  /**Create a Tensor from a vector initializer list {1, 2, 3}. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(typename detail::nested_initializer_list<1, elt_t>::type l)
      : Tensor(detail::nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a matrix braced initializer list of rows, e.g. {{1, 2, 3}, {3, 4, 5}}. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(typename detail::nested_initializer_list<2, elt_t>::type l)
      : Tensor(detail::nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a three-dimensional initializer list, e.g. {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}}. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(typename detail::nested_initializer_list<3, elt_t>::type l)
      : Tensor(detail::nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a four-dimensional initializer list. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(typename detail::nested_initializer_list<4, elt_t>::type l)
      : Tensor(detail::nested_list_initializer<elt_t>::make_tensor(l)) {}

  /**Assignment operator. Can result in both tensors sharing data.*/
  Tensor &operator=(const Tensor<elt_t> &other) = default;

  /**Assignment move operator. `other` tensor is emptied and this tensor acquires ownership of the data.*/
  Tensor &operator=(Tensor<elt_t> &&other) noexcept = default;

  /**Returns total number of elements in Tensor.*/
  size_t size() const noexcept { return dims_.total_size_t(); }
  /**Returns total number of elements in Tensor (signed type).*/
  index ssize() const noexcept { return dims_.total_size(); }
  /**Does the tensor have elements?*/
  bool is_empty() const noexcept { return ssize() == 0; }

  /**Number of Tensor indices.*/
  index rank() const noexcept { return dims_.rank(); }
  /**Return Tensor dimensions.*/
  const Dimensions &dimensions() const noexcept { return dims_; }
  /**Length of a given Tensor index, for `0 <= which < rank()`. */
  index dimension(index which) const {
    tensor_assert(rank() > which);
    tensor_assert(which >= 0);
    return dims_[which];
  }
  /**Query the size of 2nd index. Equivalent to `dimension(1)`.*/
  index columns() const { return dimension(1); }
  /**Query then size of 1st index. Equivalent to `dimension(0)`. */
  index rows() const { return dimension(0); }
  /**Query dimensions of Tensor, returned into the given pointers to variables.*/
  template <typename... index_like>
  void get_dimensions(index_like *...in) const {
    return dims_.get_values(in...);
  }

  /**Change the dimensions, while keeping the data. See \ref tensor_reshape */
  void reshape(const Dimensions &new_dimensions) {
    tensor_assert(new_dimensions.total_size() == ssize());
    dims_ = new_dimensions;
  }

  /**Return the i-th element, accessed in column major order. See \ref tensor_access*/
  inline const elt_t &operator[](index i) const noexcept {
    return cbegin()[i];
  };
  /**Return an element of a Tensor based on one or more indices. See \ref tensor_access*/
  template <typename... index_like>
  inline const elt_t &operator()(index i0, index_like... irest) const noexcept {
    return cbegin()[dims_.column_major_position(i0, irest...)];
  }

  /**Return a mutable reference to the i-th element of a Tensor, in column major order.  See \ref tensor_access*/
  inline elt_t &at_seq(index i) & { return begin()[i]; };
  /**Return a mutable reference to an element of a Tensor based on one or more indices.  See \ref tensor_access*/
  template <typename... index_like>
  inline elt_t &at(index i0, index_like... irest) & {
    return begin()[dims_.column_major_position(i0, irest...)];
  }

  /**Return the element referenced by the given indices, in column major order.*/
  inline elt_t &element_at(const Indices &i) noexcept {
    return begin()[dims_.column_major_position(i)];
  };

  /**Return the element referenced by the given indices, in column major order.*/
  inline const elt_t &element_at(const Indices &i) const noexcept {
    return cbegin()[dims_.column_major_position(i)];
  };

  /**Destructively fill this tensor with the given value. Consider using fill() instead.*/
  Tensor<elt_t> &fill_with(elt_t e);

  /**Destructively fill this tensor with zeros. Consider using zeros() instead.*/
  Tensor<elt_t> &fill_with_zeros();

  /**Destructively fill this tensor with random numbers. Consider using random() instead.*/
  Tensor<elt_t> &randomize(default_rng_t &rng = default_rng());

  /**N-dimensional tensor one or more dimensions, filled with random numbers.*/
  template <typename... index_like>
  static inline Tensor<elt_t> random(index d0,
                                     index_like... next_dimensions) noexcept {
    return random(Dimensions{d0, static_cast<index_t>(next_dimensions)...});
  }

  /**N-dimensional tensor filled with random numbers.*/
  static inline Tensor<elt_t> random(Dimensions dimensions,
                                     default_rng_t &rng = default_rng()) {
    auto output = Tensor<elt_t>::empty(dimensions);
    output.randomize_not_shared(rng);
    return output;
  }

  /**Extracts a slice from a 1D Tensor. See \ref tensor_slice */
  inline TensorView<elt_t> operator()(Range r) const {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    std::array<Range, 1> ranges{std::move(r)};
    ranges.begin()->set_dimension(ssize());
    return TensorView<elt_t>(*this, RangeSpan(ranges));
  }

  /**Extracts a slice from an N-dimensional Tensor. See \ref tensor_slice */
  template <typename... RangeLike>
  inline TensorView<elt_t> operator()(Range r1, RangeLike... rnext) const {
    std::array<Range, 1 + sizeof...(rnext)> ranges{std::move(r1),
                                                   std::move(rnext)...};
    return TensorView<elt_t>(*this, RangeSpan(ranges));
  }

  /**Extracts a slice from a 1D Tensor. See \ref tensor_slice */
  inline MutableTensorView<elt_t> at(Range r) & {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    std::array<Range, 1> ranges{std::move(r)};
    ranges.begin()->set_dimension(ssize());
    return MutableTensorView<elt_t>(*this, RangeSpan(ranges));
  }

  /**Extracts a slice from an N-dimensional Tensor. See \ref tensor_slice */
  template <typename... RangeLike>
  inline MutableTensorView<elt_t> at(Range r1, RangeLike... rnext) & {
    std::array<Range, 1 + sizeof...(rnext)> ranges{std::move(r1),
                                                   std::move(rnext)...};
    return MutableTensorView<elt_t>(*this, RangeSpan(ranges));
  }

  /**Creates a fresh new copy of this tensor, sharing memory with no other object.*/
  Tensor<elt_t> copy() const {
    auto output = Tensor<elt_t>::empty(dimensions());
    std::copy(cbegin(), cend(), output.unsafe_begin_not_shared());
    return output;
  }

  //
  // Matrix operations
  //
  /**Identity square matrix.*/
  static inline Tensor<elt_t> eye(index rows) { return eye(rows, rows); }

  /**Rectangular identity matrix.*/
  static Tensor<elt_t> eye(index rows, index cols);

  /**N-dimensional tensor with undefined values. */
  static Tensor<elt_t> empty(Dimensions &&dimensions);

  /**N-dimensional tensor with undefined values. */
  static Tensor<elt_t> empty(const Dimensions &dimensions);

  /**Empty tensor one or more dimensions, with undetermined values.*/
  template <typename... index_like>
  static inline Tensor<elt_t> empty(index_t first_dimension,
                                    index_like... nth_dimension) {
    return empty(
        Dimensions({first_dimension, static_cast<index>(nth_dimension)...}));
  }

  /**N-dimensional tensor one or more dimensions, filled with zeros.*/
  template <typename... index_like>
  static inline Tensor<elt_t> zeros(index first_dimension,
                                    index_like... next_dimensions) {
    return zeros(
        Dimensions{first_dimension, static_cast<index_t>(next_dimensions)...});
  }
  /**N-dimensional tensor filled with ones.*/
  static Tensor<elt_t> zeros(Dimensions dimensions);

  /**N-dimensional tensor one or more dimensions, filled with ones.*/
  template <typename... index_like>
  static inline Tensor<elt_t> ones(index first_dimension,
                                   index_like... next_dimensions) {
    return ones(
        Dimensions{first_dimension, static_cast<index_t>(next_dimensions)...});
  }

  /**N-dimensional tensor filled with zeros.*/
  static Tensor<elt_t> ones(Dimensions dimensions);

  /**Iterator at the beginning.
   * \todo Make begin() noexcept when we remove copy-on-write*/
#ifdef TENSOR_COPY_ON_WRITE
  iterator begin();
#else
  iterator begin() noexcept { return data_.get(); }
#endif
  /**Iterator at the beginning.*/
  const_iterator begin() const noexcept { return data_.get(); }
  /**Iterator at the beginning for const objects.*/
  const_iterator cbegin() const noexcept { return data_.get(); }
  /**Iterator at the end for const objects.*/
  const_iterator cend() const noexcept { return cbegin() + ssize(); }
  /**Iterator at the end for const objects.*/
  const_iterator end() const noexcept { return cbegin() + ssize(); }
  /**Iterator at the end.*/
  iterator end() { return begin() + ssize(); }

  iterator unsafe_begin_not_shared() noexcept { return data_.get(); }
  iterator unsafe_end_not_shared() noexcept { return data_.get() + ssize(); }

  // Only for testing purposes
  index ref_count() const noexcept { return data_.use_count(); }

  /**Take a diagonal from a tensor.*/
  Tensor<elt_t> diag(int which = 0, int ndx1 = 0, int ndx2 = -1) const {
    return take_diag(*this, which, ndx1, ndx2);
  }

  /**Create a tensor on top of data we do not own.*/
  static Tensor<elt_t> from_pointer(Dimensions dims, elt_t *data);

  /**Return a Tensor with same data and given dimensions.*/
  friend Tensor<elt_t> reshape(const Tensor<elt_t> &t, Dimensions d) {
    return Tensor<elt_t>(std::move(d), t.data_);
  }

  /**Return a RTensor with same data and given dimensions, specified separately.*/
  template <typename... index_like>
  friend inline Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1,
                                      index_like... dnext) {
    return Tensor<elt_t>(Dimensions{d1, static_cast<index>(dnext)...}, t.data_);
  }

  /**Convert a tensor to a 1D vector with the same elements.*/
  friend inline Tensor<elt_t> flatten(const Tensor<elt_t> &t) {
    return Tensor<elt_t>(Dimensions{t.ssize()}, t.data_);
  }

 private:
  shared_array<elt_t> data_{};
  Dimensions dims_{};

  /**Constructs an unitialized N-D Tensor given the dimensions.*/
  explicit Tensor(Dimensions new_dims);

  Tensor(Dimensions dimensions, shared_array<elt_t> data) noexcept;
#ifdef TENSOR_COPY_ON_WRITE
  void ensure_unique_ignore_data();
#else
  void ensure_unique_ignore_data() noexcept {}
#endif
  void randomize_not_shared(default_rng_t &rng) noexcept;
};

//
// Tensor slicing
//
template <typename elt_t>
class TensorView {
 public:
  TensorView() = delete;
  ~TensorView() = default;
  TensorView(const TensorView<elt_t> &other) = default;
  TensorView(TensorView<elt_t> &&other) = default;
  TensorView &operator=(const TensorView &) = delete;
  TensorView &operator=(TensorView &&) = delete;

  TensorView(const Tensor<elt_t> &tensor, RangeSpan ranges)
      : tensor_(tensor),
        dims_(ranges.get_dimensions(tensor.dimensions())),
        range_iterator_begin_(RangeIterator::begin(ranges)) {}

  size_t size() const { return static_cast<size_t>(dims_.total_size()); }
  index ssize() const { return dims_.total_size(); }
  const Dimensions &dimensions() const { return dims_; }

  TensorConstIterator<elt_t> begin() const {
    return TensorConstIterator<elt_t>(range_iterator_begin_, tensor_.begin());
  }
  TensorConstIterator<elt_t> end() const {
    return TensorConstIterator<elt_t>(range_iterator_begin_.make_end_iterator(),
                                      tensor_.begin());
  }

  Tensor<elt_t> copy() const {
    auto output = Tensor<elt_t>::empty(dimensions());
    begin().copy_to_contiguous_iterator(output.unsafe_begin_not_shared());
    return output;
  }

  [[deprecated]] Tensor<elt_t> clone() const { return copy(); }

  operator Tensor<elt_t>() const { return copy(); }

 private:
  const Tensor<elt_t> &tensor_;
  Dimensions dims_;
  RangeIterator range_iterator_begin_;
};

template <typename elt_t>
class MutableTensorView {
 public:
  MutableTensorView() = delete;
  ~MutableTensorView() = default;
  MutableTensorView(const MutableTensorView<elt_t> &other) = default;
  MutableTensorView(MutableTensorView<elt_t> &&other) = default;

  MutableTensorView(Tensor<elt_t> &tensor, RangeSpan ranges)
      : tensor_(tensor),
        dims_(ranges.get_dimensions(tensor.dimensions())),
        range_iterator_begin_(RangeIterator::begin(ranges)) {}

  MutableTensorView &operator=(const TensorView<elt_t> &t) {
    tensor_assert(dimensions() == t.dimensions());
    begin().copy_from(t.begin());
    return *this;
  }
  MutableTensorView &operator=(const Tensor<elt_t> &t) {
    tensor_assert(dimensions() == t.dimensions());
    begin().copy_from_contiguous_iterator(t.begin());
    return *this;
  }
  MutableTensorView &operator=(elt_t v) {
    begin().fill(v);
    return *this;
  }

  size_t size() const { return static_cast<size_t>(dims_.total_size()); }
  index ssize() const { return dims_.total_size(); }
  const Dimensions &dimensions() const { return dims_; }

  TensorIterator<elt_t> begin() {
    return TensorIterator<elt_t>(range_iterator_begin_, tensor_.begin());
  }
  TensorIterator<elt_t> end() {
    return TensorIterator<elt_t>(range_iterator_begin_.make_end_iterator(),
                                 tensor_.begin());
  }
  TensorConstIterator<elt_t> begin() const {
    return TensorConstIterator<elt_t>(range_iterator_begin_, tensor_.begin());
  }
  TensorConstIterator<elt_t> end() const {
    return TensorConstIterator<elt_t>(range_iterator_begin_.make_end_iterator(),
                                      tensor_.begin());
  }
  Tensor<elt_t> copy() const {
    auto output = Tensor<elt_t>::empty(dimensions());
    std::copy(begin(), end(), output.unsafe_begin_not_shared());
    return output;
  }

 private:
  Tensor<elt_t> &tensor_;
  Dimensions dims_;
  RangeIterator range_iterator_begin_;
};

extern template class Tensor<double>;
extern template class TensorView<double>;
extern template class MutableTensorView<double>;
#ifdef DOXYGEN_ONLY
/** Real Tensor with elements of type "double". */
struct RTensor : public Tensor<double> {};
#else
using RTensor = Tensor<double>;
#endif

extern template class Tensor<cdouble>;
extern template class TensorView<cdouble>;
extern template class MutableTensorView<cdouble>;
/** Complex Tensor with elements of type "cdouble". */
#ifdef DOXYGEN_ONLY
struct CTensor : public Tensor<cdouble> {}
#else
using CTensor = Tensor<cdouble>;
#endif

}  // namespace tensor

/* @} */

#endif  // TENSOR_TENSOR_BASE_H
