// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <tensor/numbers.h>
#include <tensor/vector.h>
#include <tensor/indices.h>

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
  Tensor();

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

  index size() const { return data_.size(); }

  /**Return tensor dimensions.*/
  const Indices &dims() { return dims_; }

  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator begin_const() const { return data_.begin_const(); }
  const_iterator end_const() const { return data_.end_const(); }
  const_iterator end() const { return data_.end(); }
  iterator end() { return data_.end(); }
  
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

//
// ALGEBRA
//

//////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//
#ifdef TENSOR_LOAD_IMPL
#include <tensor/detail/tensor_base.hpp>
#include <tensor/detail/tensor_reshape.hpp>
#endif

//////////////////////////////////////////////////////////////////////
// EXPLICIT INSTANTIATIONS
//

extern template class Tensor<double>;
typedef Tensor<double> RTensor;

extern template class Tensor<cdouble>;
typedef Tensor<cdouble> CTensor;

} // namespace

#endif // !TENSOR_H
