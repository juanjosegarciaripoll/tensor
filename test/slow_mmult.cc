// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <cassert>
#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor_test {

  using namespace tensor;
  using tensor::index;

  template<typename n1, typename n2>
  Tensor<typename Binop<n1,n2>::type>
  fold_22_12(const Tensor<n1> &A, const Tensor<n2> &B)
  {
    typedef typename Binop<n1,n2>::type n3;
    index a1, a2, b1, b2;
    A.get_dimensions(&a1, &a2);
    B.get_dimensions(&b1, &b2);
    assert(a2==b1);

    Tensor<n3> output(a1,b2);

    for (index i = 0; i < a1; i++) {
      for (index k = 0; k < b2; k++) {
        n3 x = number_zero<n3>();
        for (index j = 0; j < a2; j++) {
          x += A(i,j) * B(j,k);
        }
        output.at(i,k) = x;
      }
    }
    return output;
  }

} // namespace tensor_test
