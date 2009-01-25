// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <tensor/numbers.h>
#include <tensor/vector.h>
#include <tensor/indices.h>
#include <tensor/detail/functional.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// BASE CLASS
//

/*!\addtogroup Tensors*/
/*!@{*/
/**An N-dimensional array of numbers. Tensors are much like Matlab's arrays. They can
   store numbers, be accessed with one, two or more indices, be reshaped,
   extract and put elements, etc.

   \anchor tensor_index The total number of dimensions or indices in a tensor
   is given by the function rank(). However, for many operations we do not
   want to care about this number, but rather refer to the first index, or to
   the last, or to the first-to-last... For this reason, some functions allow
   negative dimension index: 0 is the first index, 1 is the second, etc, and
   then -1 is interpreted as the last index, -2, as the first to one, etc.

   The reason why the first index is labeled 0 is because of consistency with
   the C/C++ indexing of arrays.

   \anchor matrices In most computer algebra environments, a tensor with two
   indices is referred as a matrix and it is treated specially. In MPSLIB
   matrices are no special citizens, but nevertheless we provide some
   functions, such as columns(), rows(), etc, which do what expected and only
   work with 2D arrays.

   \anchor Copy Our implementation is such that when a tensor is copied, the
   data is not copied unless either the original tensor or the copy are
   modified. For instance, take the following piece of code,
   \code
   ...
   Tensor a = b; //[1]
   ...
   a(1) = 2; //[2]
   ...
   \endcode
   In line [1] just a pointer is copied from b to a. However, in line [2] just
   before modifying the data, we make a copy of it. In the end, b still has
   the original data, and a the modified copy.
*/
template<typename elt>
class Tensor {
 public:
  typedef elt elt_t;
  typedef elt_t *iterator;
  typedef const elt_t *const_iterator;

  /**Constructs an empty tensor.*/
  Tensor() : data_(), dims_() {}

  /**Constructs an unitialized N-D tensor given the dimensions.*/
  explicit Tensor(const Indices &new_dims);

  /**Consturcts an N-D tensor with given initial data.*/
  Tensor(const Indices &new_dims, const Tensor<elt_t> &data);

  /**Optimized copy constructor (See \ref Copy "Optimal copy").*/
  Tensor(const Tensor &other) : data_(other.data_), dims_(other.dims_) {}

  /**Build a 1D tensor given the size and the raw C data.*/
  explicit Tensor(index length);
  /**Build a matrix.*/
  Tensor(index rows, index cols);
  /**Build a 3D tensor.*/
  Tensor(index d1, index d2, index d3);
  /**Build a 4D tensor.*/
  Tensor(index d1, index d2, index d3, index d4);
  /**Build a 5D tensor.*/
  Tensor(index d1, index d2, index d3, index d4, index d5);
  /**Build a 6D tensor.*/
  Tensor(index d1, index d2, index d3, index d4, index d5, index d6);

  /**Returns total number of elements in tensor.*/
  index size() const { return data_.size(); }

  /**Number of tensor indices.*/
  int rank() const { return dims_.size(); }
  /**Return tensor dimensions.*/
  const Indices &dimensions() const { return dims_; }
  /**Length of a given tensor index.*/
  index dimension(int which) const;
  /**Query dimensions of 1D tensor.*/
  void get_dimensions(index *length) const;
  /**Query dimensions of 2D tensor.*/
  void get_dimensions(index *rows, index *cols) const;
  /**Query dimensions of 3D tensor.*/
  void get_dimensions(index *d0, index *d1, index *d2) const;
  /**Query dimensions of 4D tensor.*/
  void get_dimensions(index *d0, index *d1, index *d2, index *d3) const;
  /**Query dimensions of 5D tensor.*/
  void get_dimensions(index *d0, index *d1, index *d2, index *d3, index *d4) const;
  /**Query dimensions of 6D tensor.*/
  void get_dimensions(index *d0, index *d1, index *d2, index *d3, index *d4, index *d5) const;
  /**Query size of 2nd index.*/
  index columns() const { return dimension(1); }
  /**Query size of 1st index. */
  index rows() const { return dimension(0); }

  /**Change the dimensions, while keeping the data. */
  void reshape(const Indices &new_dims);

  /**Return element in linear order.*/
  const elt_t &operator[](index i) const;
  /**Return element of 1D tensor.*/
  const elt_t &operator()(index i) const;
  /**Return element of 2D tensor.*/
  const elt_t &operator()(index row, index col) const;
  /**Return element of 3D tensor.*/
  const elt_t &operator()(index d0, index d1, index d2) const;
  /**Return element of 4D tensor.*/
  const elt_t &operator()(index d0, index d1, index d2, index d3) const;
  /**Return element of 5D tensor.*/
  const elt_t &operator()(index d0, index d1, index d2, index d3, index d4) const;
  /**Return element of 6D tensor.*/
  const elt_t &operator()(index d0, index d1, index d2, index d3, index d4, index d5w) const;

  /**Return mutable reference to element of a tensor.*/
  elt_t &at(index i);
  /**Return mutable reference to element of 2D tensor.*/
  elt_t &at(index row, index col);
  /**Return mutable reference to element of 3D tensor.*/
  elt_t &at(index d1, index d2, index d3);
  /**Return mutable reference to element of 4D tensor.*/
  elt_t &at(index d1, index d2, index d3, index d4);
  /**Return mutable reference to element of 5D tensor.*/
  elt_t &at(index d1, index d2, index d3, index d4, index d5);
  /**Return mutable reference to element of 6D tensor.*/
  elt_t &at(index d1, index d2, index d3, index d4, index d5, index d6);

  /**Fill with an element.*/
  void fill_with(const elt_t &e);
  /**Fill with zeros.*/
  void fill_with_zeros() { fill_with(number_zero<elt_t>()); }
  /**Fills with random numbers.*/
  void randomize();

  //
  // Matrix operations
  //
  /**Identity matrix.*/
  static Tensor<elt_t> eye(index rows) { return eye(rows, rows); }
  /**Rectangular identity matrix.*/
  static Tensor<elt_t> eye(index rows, index cols);
  /**Matrix of zeros.*/
  static Tensor<elt_t> zeros(index rows) { return zeros(rows, rows); }
  /**Matrix of ones.*/
  static Tensor<elt_t> zeros(index rows, index cols);
  /**Matrix of ones.*/
  static Tensor<elt_t> ones(index rows) { return ones(rows, rows); }
  /**Matrix of ones.*/
  static Tensor<elt_t> ones(index rows, index cols);

  /**Iterator at the beginning.*/
  iterator begin() { return data_.begin(); }
  /**Iterator at the beginning.*/
  const_iterator begin() const { return data_.begin_const(); }
  /**Iterator at the beginning for const objects.*/
  const_iterator begin_const() const { return data_.begin_const(); }
  /**Iterator at the end for const objects.*/
  const_iterator end_const() const { return data_.end_const(); }
  /**Iterator at the end for const objects.*/
  const_iterator end() const { return data_.end_const(); }
  /**Iterator at the end.*/
  iterator end() { return data_.end(); }

  // Only for testing purposes
  int ref_count() const { return data_.ref_count(); }

 private:
  Vector<elt_t> data_;
  Indices dims_;
};

//////////////////////////////////////////////////////////////////////
// TENSOR GENERIC OPERATIONS
//
//
// RESHAPING
//

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, const Indices &new_dims);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index length);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index rows, index columns);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
		      index d4);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
		      index d4, index d5);

/**Return a tensor with same data and given dimensions.*/
template<typename elt_t>
Tensor<elt_t> reshape(const Tensor<elt_t> &t, index d1, index d2, index d3,
		      index d4, index d5, index d6);

//////////////////////////////////////////////////////////////////////
// ALGEBRA
//
//
// Unary operations
//
template<typename t>
Tensor<t> operator-(const Tensor<t> &t);

//
// Binary operations
//
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const Tensor<t1> &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const Tensor<t1> &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const Tensor<t1> &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const Tensor<t1> &a, const Tensor<t2> &b);
template<typename t1, typename t2>
bool operator==(const Tensor<t1> &a, const Tensor<t2> &b);

template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const Tensor<t1> &a, const t2 &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const Tensor<t1> &a, const t2 &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const Tensor<t1> &a, const t2 &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const Tensor<t1> &a, const t2 &b);

template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const t1 &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const t1 &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const t1 &a, const Tensor<t2> &b);
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const t1 &a, const Tensor<t2> &b);

} // namespace tensor

//////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//
#ifdef TENSOR_LOAD_IMPL
#include <tensor/detail/tensor_base.hpp>
#include <tensor/detail/tensor_matrix.hpp>
#endif
#include <tensor/detail/tensor_reshape.hpp>
#include <tensor/detail/tensor_ops.hpp>

//////////////////////////////////////////////////////////////////////
// EXPLICIT INSTANTIATIONS
//

namespace tensor {

  extern template class Tensor<double>;
  typedef Tensor<double> RTensor;

  double norm0(const RTensor &r);

  RTensor abs(const RTensor &t);
  RTensor cos(const RTensor &t);
  RTensor sin(const RTensor &t);
  RTensor tan(const RTensor &t);
  RTensor cosh(const RTensor &t);
  RTensor sinh(const RTensor &t);
  RTensor tanh(const RTensor &t);
  RTensor exp(const RTensor &t);

  const RTensor diag(const RTensor &d, int which, int rows, int cols);
  const RTensor diag(const RTensor &d, int which = 0);
  double trace(const RTensor &d);

  const RTensor permute(const RTensor &a, index ndx1 = 0, index ndx2 = -1);
  const RTensor transpose(const RTensor &a);
  inline const RTensor adjoint(const RTensor &a) { return transpose(a); }

  const RTensor fold(const RTensor &A, int ndx1, const RTensor &b, int ndx2);
  const RTensor mmult(const RTensor &A, const RTensor &b);

  extern template class Tensor<cdouble>;
  typedef Tensor<cdouble> CTensor;

  double norm0(const CTensor &r);

  const CTensor to_complex(const RTensor &r);
  inline const CTensor to_complex(const CTensor &r) { return r; }
  const CTensor to_complex(const RTensor &r, const RTensor &i);

  RTensor abs(const CTensor &t);
  CTensor cos(const CTensor &t);
  CTensor sin(const CTensor &t);
  CTensor tan(const CTensor &t);
  CTensor cosh(const CTensor &t);
  CTensor sinh(const CTensor &t);
  CTensor tanh(const CTensor &t);
  CTensor exp(const CTensor &t);

  const CTensor diag(const CTensor &d, int which, int rows, int cols);
  const CTensor diag(const CTensor &d, int which = 0);
  cdouble trace(const CTensor &d);

  const CTensor permute(const CTensor &a, index ndx1 = 0, index ndx2 = -1);
  const CTensor transpose(const CTensor &a);
  const CTensor adjoint(const CTensor &a);

  const CTensor fold(const CTensor &A, int ndx1, const CTensor &b, int ndx2);
  const CTensor fold(const RTensor &A, int ndx1, const CTensor &b, int ndx2);
  const CTensor fold(const CTensor &A, int ndx1, const RTensor &b, int ndx2);

  const CTensor mmult(const CTensor &A, const CTensor &b);
  const CTensor mmult(const RTensor &A, const CTensor &b);
  const CTensor mmult(const CTensor &A, const RTensor &b);

} // namespace tensor

#endif // !TENSOR_H
