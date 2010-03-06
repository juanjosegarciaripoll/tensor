// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_GEN_H
#define TENSOR_GEN_H

#include <tensor/numbers.h>
#include <tensor/vector.h>

namespace tensor {

  /** Compile-time generator of vectors, tensors and indices. This placeholder
      can be used to statically create arrays of data that can then be coerced
      to Vector, Tensor and Indices types. There are three generators tensor::igen,
      tensor::rgen and tensor::cgen of types ListGenerator<t> with t = tensor::index,
      double and cdouble, which can be used as shown in \ref sec_tensor_gen
\code
RTensor a = rgen << 2 << 3 << 4;
Indices d = igen << 1 << 2 << 3 << 5;
\endcode

  */
  template<typename elt_t> class ListGenerator {};

  /** Placeholder for statically creating a vector of indices. For example
\code
Indices d = igen << 2 << 2;
RTensor a(igen << 1 << 3);
\endcode
\sa ListGenerator */
  extern const ListGenerator<index> igen;
  /** Placeholder for statically creating a vector of reals.
\code
RTensor a = rgen << 1 << 3;
\endcode
\sa ListGenerator */
  extern const ListGenerator<double> rgen;
  /** Placeholder for statically creating a vector of complex numbers.
\code
CTensor a = cgen << 1 << 3;
\endcode
\sa ListGenerator */
  extern const ListGenerator<cdouble> cgen;

  template<typename elt_t, size_t n>
  class StaticVector {
  public:
    StaticVector(const StaticVector<elt_t,n-1> &other, elt_t x) :
      inner(other), extra(x)
    {};
    operator Vector<elt_t>() const {
      Vector<elt_t> output(n);
      push(output.begin());
      return output;
    }
    void push(elt_t *v) const {
      inner.push(v);
      v[n-1] = extra;
    }
    index size() const {
      return n;
    }
  protected:
    StaticVector<elt_t,n-1> inner;
    elt_t extra;
  };

  template<typename elt_t>
  class StaticVector<elt_t,(size_t)1> {
  public:
    StaticVector(elt_t x) :
      extra(x)
    {};
    operator Vector<elt_t>() const {
      Vector<elt_t> output(1);
      push(output.begin());
      return output;
    }
    void push(elt_t *v) const {
      v[0] = extra;
    }
    index size() const {
      return 1;
    }
  private:
    elt_t extra;
  };

  template<typename t1, typename t2>
  StaticVector<t1,1> operator<<(const ListGenerator<t1> &g, const t2 x) {
    return StaticVector<t1,1>(x);
  }

  template<typename t1, typename t2, size_t n>
  StaticVector<t1,n+1> operator<<(const StaticVector<t1,n> &g, const t2 x) {
    return StaticVector<t1,n+1>(g,x);
  }

} // namespace tensor

#endif // TENSOR_GEN_H
