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
  typedef elt elt_t;
  /** The type of this tensor's elements */
  typedef elt value_type;
  /** Random access iterator type */
  typedef elt_t *iterator;
  /** Random access iterator type to const */
  typedef const elt_t *const_iterator;
#ifdef TENSOR_COPY_ON_WRITE
  /** Container for this tensor's data */
  typedef Vector<elt_t> vector_type;
#else
  /** Container for this tensor's data */
  typedef SimpleVector<elt_t> vector_type;
#endif

  /**Constructs an empty Tensor.*/
  Tensor() = default;

  /**Constructs an unitialized N-D Tensor given the dimensions.*/
  explicit Tensor(const Dimensions &new_dims)
      : data_(static_cast<size_t>(new_dims.total_size())), dims_(new_dims){};

  /**Constructs an N-D Tensor with given initial data.*/
  Tensor(const Dimensions &new_dims, const Tensor<elt_t> &other)
      : data_(other.data_), dims_(new_dims) {
    tensor_assert(dims_.total_size() == ssize());
  }

  /**Constructs a 1-D Tensor from a vector.*/
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(const vector_type &data) : data_(data), dims_{data_.size()} {}

  /**Constructs a 1-D Tensor from a vector (move version for temporaries).*/
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(vector_type &&data) noexcept
      : data_(std::move(data)), dims_({data_.size()}) {}

  /**Constructs a 1-D Tensor from a vector.*/
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Tensor(const std::vector<elt_t> &data)
      : data_(data.size()), dims_{static_cast<index>(data.size())} {
    std::copy(data.begin(), data.end(), begin());
  }

  /**Optimized copy constructor.*/
  Tensor(const Tensor &other) = default;

  /**Optimized move constructor. */
  Tensor(Tensor &&other) = default;

  /**Implicit coercion and transformation to a different type. */
  // NOLINTNEXTLINE(*-explicit-constructor)
  template <typename e2>
  // cppcheck-suppress noExplicitConstructor
  Tensor(const Tensor<e2> &other)
      : data_(other.size()), dims_(other.dimensions()) {
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

#if 0
  /**Build a 1D Tensor or vector.*/
  explicit Tensor(index length);
  /**Build a 2D Tensor or matrix.*/
  Tensor(index rows, index cols);
  /**Build a 3D Tensor.*/
  Tensor(index d1, index d2, index d3);
  /**Build a 4D Tensor.*/
  Tensor(index d1, index d2, index d3, index d4);
  /**Build a 5D Tensor.*/
  Tensor(index d1, index d2, index d3, index d4, index d5);
  /**Build a 6D Tensor.*/
  Tensor(index d1, index d2, index d3, index d4, index d5, index d6);
#endif

  /**Explicit copy of this tensor's data as a vector.*/
  explicit operator vector_type() const { return data_; }

  /**Assignment operator. Can result in both tensors sharing data.*/
  Tensor &operator=(const Tensor<elt_t> &other) = default;

  /**Assignment move operator. `other` tensor is emptied and this tensor acquires ownership of the data.*/
  Tensor &operator=(Tensor<elt_t> &&other) = default;

  /**Returns total number of elements in Tensor.*/
  size_t size() const noexcept { return data_.size(); }
  /**Returns total number of elements in Tensor (signed type).*/
  index ssize() const noexcept { return data_.ssize(); }
  /**Does the tensor have elements?*/
  bool is_empty() const noexcept { return size() == 0; }

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
  inline const elt_t &operator[](index i) const noexcept { return data_[i]; };
  /**Return an element of a Tensor based on one or more indices. See \ref tensor_access*/
  template <typename... index_like>
  inline const elt_t &operator()(index i0, index_like... irest) const noexcept {
    return data_[dims_.column_major_position(i0, irest...)];
  }

  /**Return a mutable reference to the i-th element of a Tensor, in column major order.  See \ref tensor_access*/
  inline elt_t &at_seq(index i) { return data_.at(i); };
  /**Return a mutable reference to an element of a Tensor based on one or more indices.  See \ref tensor_access*/
  template <typename... index_like>
  inline elt_t &at(index i0, index_like... irest) {
    return data_.at(dims_.column_major_position(i0, irest...));
  }

  /**Destructively full this tensor with the given value. Consider using fill() instead.*/
  Tensor<elt_t> &fill_with(const elt_t &e) noexcept {
    std::fill(begin(), end(), e);
    return *this;
  }
  /**Destructively fill this tensor with zeros. Consider using zeros() instead.*/
  Tensor<elt_t> &fill_with_zeros() noexcept {
    return fill_with(number_zero<elt_t>());
  }
  /**Destructively fill this tensor with random numbers. Consider using random() instead.*/
  Tensor<elt_t> &randomize() noexcept {
    std::generate(this->begin(), this->end(),
                  []() -> elt_t { return rand<elt_t>(); });
    return *this;
  }

  /**N-dimensional tensor one or more dimensions, filled with random numbers.*/
  template <typename... index_like>
  static inline Tensor<elt_t> random(index d0,
                                     index_like... next_dimensions) noexcept {
    return Tensor<elt_t>::empty(d0, next_dimensions...).randomize();
  }

  /**N-dimensional tensor filled with random numbers.*/
  static inline Tensor<elt_t> random(const Dimensions &dimensions) noexcept {
    return Tensor<elt_t>(dimensions).randomize();
  };

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
  inline MutableTensorView<elt_t> at(Range r) {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    std::array<Range, 1> ranges{std::move(r)};
    ranges.begin()->set_dimension(ssize());
    return MutableTensorView<elt_t>(*this, RangeSpan(ranges));
  }

  /**Extracts a slice from an N-dimensional Tensor. See \ref tensor_slice */
  template <typename... RangeLike>
  inline MutableTensorView<elt_t> at(Range r1, RangeLike... rnext) {
    std::array<Range, 1 + sizeof...(rnext)> ranges{std::move(r1),
                                                   std::move(rnext)...};
    return MutableTensorView<elt_t>(*this, RangeSpan(ranges));
  }

  //
  // Matrix operations
  //
  /**Identity square matrix.*/
  static inline Tensor<elt_t> eye(index rows) { return eye(rows, rows); }
  /**Rectangular identity matrix.*/
  static Tensor<elt_t> eye(index rows, index cols) {
    auto output = empty(rows, cols);
    output.fill_with_zeros();
    for (index i = 0; i < rows && i < cols; ++i) {
      output.at(i, i) = number_one<elt_t>();
    }
    return output;
  }

  /**N-dimensional tensor with undefined values. */
  static inline Tensor<elt_t> empty(const Dimensions &dimensions) {
    return Tensor<elt_t>(dimensions);
  }

  /**Empty tensor one or more dimensions, with undetermined values.*/
  template <typename... index_like>
  static inline Tensor<elt_t> empty(index_like... nth_dimension) {
    return Tensor<elt_t>(Dimensions({static_cast<index>(nth_dimension)...}));
  }

  /**N-dimensional tensor one or more dimensions, filled with zeros.*/
  template <typename... index_like>
  static inline Tensor<elt_t> zeros(index first_dimension,
                                    index_like... next_dimensions) {
    return Tensor::empty(first_dimension, next_dimensions...).fill_with_zeros();
  }
  /**N-dimensional tensor filled with ones.*/
  static inline Tensor<elt_t> zeros(const Dimensions &dimensions) {
    return Tensor<elt_t>(dimensions).fill_with_zeros();
  }

  /**N-dimensional tensor one or more dimensions, filled with ones.*/
  template <typename... index_like>
  static inline Tensor<elt_t> ones(index first_dimension,
                                   index_like... next_dimensions) {
    return Tensor::empty(first_dimension, next_dimensions...)
        .fill_with(number_one<elt_t>());
  }
  /**N-dimensional tensor filled with zeros.*/
  static inline Tensor<elt_t> ones(const Dimensions &dimensions) {
    return Tensor<elt_t>(dimensions).fill_with(number_one<elt_t>());
  };

  /**Iterator at the beginning.
   * \todo Make begin() noexcept when we remove copy-on-write*/
  iterator begin() { return data_.begin(); }
  /**Iterator at the beginning.*/
  const_iterator begin() const noexcept { return data_.cbegin(); }
  /**Iterator at the beginning for const objects.*/
  const_iterator cbegin() const noexcept { return data_.cbegin(); }
  /**Iterator at the end for const objects.*/
  const_iterator cend() const noexcept { return data_.cend(); }
  /**Iterator at the end for const objects.*/
  const_iterator end() const noexcept { return data_.cend(); }
  /**Iterator at the end.*/
  iterator end() { return data_.end(); }

  // Only for testing purposes
  index ref_count() const noexcept { return data_.ref_count(); }

  /**Take a diagonal from a tensor.*/
  const Tensor<elt_t> diag(int which = 0, int ndx1 = 0, int ndx2 = -1) {
    return take_diag(*this, which, ndx1, ndx2);
  }

 private:
  vector_type data_{};
  Dimensions dims_{};
};
//
// Tensor slicing
//
template <typename elt_t>
class TensorView {
 public:
  TensorView() = delete;
  TensorView(const TensorView<elt_t> &other) = default;
  TensorView(TensorView<elt_t> &&other) = default;

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
    Tensor<elt_t> output(dimensions());
    begin().copy_to_contiguous_iterator(output.begin());
    return output;
  }

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
  MutableTensorView(const MutableTensorView<elt_t> &other) = default;
  MutableTensorView(MutableTensorView<elt_t> &&other) = default;

  MutableTensorView(Tensor<elt_t> &tensor, RangeSpan ranges)
      : tensor_(tensor),
        dims_(ranges.get_dimensions(tensor.dimensions())),
        range_iterator_begin_(RangeIterator::begin(ranges)) {}

  void operator=(const TensorView<elt_t> &t) {
    tensor_assert(
        verify_tensor_dimensions_match(this->dimensions(), t.dimensions()));
    begin().copy_from(t.begin());
  }
  void operator=(const Tensor<elt_t> &t) {
    tensor_assert(
        verify_tensor_dimensions_match(this->dimensions(), t.dimensions()));
    begin().copy_from_contiguous_iterator(t.begin());
  }
  void operator=(elt_t v) { begin().fill(v); }

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
    Tensor<elt_t> output(dimensions());
    std::copy(begin(), end(), output.begin());
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
typedef Tensor<double> RTensor;
#endif

extern template class Tensor<cdouble>;
extern template class TensorView<cdouble>;
extern template class MutableTensorView<cdouble>;
/** Complex Tensor with elements of type "cdouble". */
#ifdef DOXYGEN_ONLY
struct CTensor : public Tensor<cdouble> {}
#else
typedef Tensor<cdouble> CTensor;
#endif

}  // namespace tensor

/* @} */

#endif  // TENSOR_TENSOR_BASE_H
