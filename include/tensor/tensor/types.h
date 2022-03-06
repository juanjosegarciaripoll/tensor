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
#include <tensor/initializer.h>
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
   \see \ref sec_tensor
*/
template <typename elt>
class Tensor {
 public:
  typedef elt elt_t;
  typedef elt value_type;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;
#ifdef TENSOR_COPY_ON_WRITE
  typedef Vector<elt_t> vector_type;
#else
  typedef SimpleVector<elt_t> vector_type;
#endif

  /**Constructs an empty Tensor.*/
  Tensor() noexcept = default;

  /**Constructs an unitialized N-D Tensor given the dimensions.*/
  explicit Tensor(const Dimensions &new_dims)
      : data_(static_cast<size_t>(new_dims.total_size())), dims_(new_dims){};

  /**Constructs an N-D Tensor with given initial data.*/
  Tensor(const Dimensions &new_dims, const Tensor<elt_t> &other)
      : data_(other.data_), dims_(new_dims) {
    tensor_assert(dims_.total_size() == ssize());
  }

  /**Constructs a 1-D Tensor from a vector.*/
  Tensor(const vector_type &data) : data_(data), dims_{data_.size()} {}

  /**Constructs a 1-D Tensor from a vector (move version for temporaries).*/
  Tensor(vector_type &&data) noexcept
      : data_(std::move(data)), dims_({data_.size()}) {}

  /**Constructs a 1-D Tensor from a vector.*/
  Tensor(const std::vector<elt_t> &data)
      : data_(data.size()), dims_{static_cast<index>(data.size())} {
    std::copy(data.begin(), data.end(), begin());
  }

  /**Optimized copy constructor (See \ref Copy "Optimal copy").*/
  Tensor(const Tensor &other) = default;

  /**Optimized move constructor. */
  Tensor(Tensor &&other) noexcept = default;

  /**Implicit coercion. */
  template <typename e2>
  Tensor(const Tensor<e2> &other)
      : data_(other.size()), dims_(other.dimensions()) {
    std::copy(other.begin(), other.end(), begin());
  }

  /**Create a Tensor from a vector initializer list {1, 2, 3}. */
  Tensor(typename nested_initializer_list<1, elt_t>::type l)
      : Tensor(nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a matrix braced initializer list of rows, e.g. {{1, 2, 3}, {3, 4, 5}}. */
  Tensor(typename nested_initializer_list<2, elt_t>::type l)
      : Tensor(nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a three-dimensional initializer list, e.g. {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}}. */
  Tensor(typename nested_initializer_list<3, elt_t>::type l)
      : Tensor(nested_list_initializer<elt_t>::make_tensor(l)) {}
  /**Create a Tensor from a four-dimensional initializer list. */
  Tensor(typename nested_initializer_list<4, elt_t>::type l)
      : Tensor(nested_list_initializer<elt_t>::make_tensor(l)) {}

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

  explicit operator vector_type() const { return data_; }

  /**Assignment operator.*/
  Tensor &operator=(const Tensor<elt_t> &other) = default;

  /**Assignment move operator.*/
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
  /**Length of a given Tensor index.*/
  index dimension(index which) const {
    tensor_assert(rank() > which);
    tensor_assert(which >= 0);
    return dims_[which];
  }
  /**Query the size of 2nd index.*/
  index columns() const { return dimension(1); }
  /**Query then size of 1st index. */
  index rows() const { return dimension(0); }
  /**Query dimensions of Tensor, returned into the given pointers.*/
  template <typename... index_like>
  void get_dimensions(index_like *...in) const {
    return dims_.get_values(in...);
  }

  /**Change the dimensions, while keeping the data. */
  void reshape(const Dimensions &new_dimensions) {
    tensor_assert(new_dimensions.total_size() == ssize());
    dims_ = new_dimensions;
  }

  /**Return the i-th element, accessed in column major order.*/
  inline const elt_t &operator[](index i) const noexcept { return data_[i]; };
  /**Return an element of a Tensor based on one or more indices.*/
  template <typename... index_like>
  inline const elt_t &operator()(index i0, index_like... irest) const noexcept {
    return data_[dims_.column_major_position(i0, irest...)];
  }

  /**Return a mutable reference to the i-th element of a Tensor, in column major order.*/
  inline elt_t &at_seq(index i) { return data_.at(i); };
  /**Return a mutable reference to an element of a Tensor based on one or more indices.*/
  template <typename... index_like>
  inline elt_t &at(index i0, index_like... irest) {
    return data_.at(dims_.column_major_position(i0, irest...));
  }

  /**Fill with an element.*/
  Tensor<elt_t> &fill_with(const elt_t &e) noexcept {
    std::fill(begin(), end(), e);
    return *this;
  }
  /**Fill with zeros.*/
  Tensor<elt_t> &fill_with_zeros() noexcept {
    return fill_with(number_zero<elt_t>());
  }
  /**Fills with random numbers.*/
  Tensor<elt_t> &randomize() noexcept {
    for (auto &x : *this) {
      x = rand<elt_t>();
    }
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

  /**Extracts a slice from a 1D Tensor. */
  inline TensorView<elt_t> operator()(Range r) const {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    std::array<Range, 1> ranges{std::move(r)};
    ranges.begin()->set_dimension(ssize());
    return TensorView<elt_t>(*this, ranges);
  }

  /**Extracts a slice from an N-dimensional Tensor. */
  template <typename... RangeLike>
  inline TensorView<elt_t> operator()(Range r1, RangeLike... rnext) const {
    std::array<Range, 1 + sizeof...(rnext)> ranges{std::move(r1),
                                                   std::move(rnext)...};
    return TensorView<elt_t>(*this, ranges);
  }

  /**Extracts a slice from a 1D Tensor. */
  inline MutableTensorView<elt_t> at(Range r) {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    std::array<Range, 1> ranges{std::move(r)};
    ranges.begin()->set_dimension(ssize());
    return MutableTensorView<elt_t>(*this, ranges);
  }

  /**Extracts a slice from an N-dimensional Tensor. */
  template <typename... RangeLike>
  inline MutableTensorView<elt_t> at(Range r1, RangeLike... rnext) {
    std::array<Range, 1 + sizeof...(rnext)> ranges{std::move(r1),
                                                   std::move(rnext)...};
    return MutableTensorView<elt_t>(*this, ranges);
  }

  //
  // Matrix operations
  //
  /**Identity matrix.*/
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

  /**N-dimensional tensor with undefined values. */
  static inline Tensor<elt_t> empty(const Indices &dimensions) {
    return Tensor<elt_t>(Dimensions(dimensions));
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

  /* TODO: Make begin() noexcept when we remove copy-on-write */
  /**Iterator at the beginning.*/
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
  operator Tensor<elt_t>() const {
    Tensor<elt_t> output(dimensions());
    std::copy(begin(), end(), output.begin());
    return output;
  }

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
    std::copy(begin(), end(), output.begin());
    return output;
  }

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
    /* TODO: ensure matching dimensions */
    tensor_assert(t.size() == size());
    std::copy(t.begin(), t.end(), begin());
  }
  void operator=(const Tensor<elt_t> &t) {
    /* TODO: ensure matching dimensions */
    tensor_assert(t.size() == size());
    std::copy(t.begin(), t.end(), begin());
  }
  void operator=(elt_t v) { std::fill(begin(), end(), v); }

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