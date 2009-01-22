// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

  template<typename n>
  void permute_12(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3)
  {
    // In an abstract sense b(j,i,k) = a(i,j,k)
    // Both tensors are stored in row-major order. So for A we have
    // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
    //
    typename Tensor<n>::const_iterator ijk_a = a.begin();
    typename Tensor<n>::iterator k_b = b.begin();
    for (; a3; a3--, k_b += (a1*a2)) {
      typename Tensor<n>::iterator jk_b = k_b;
      for (index j = a2; j--; jk_b++) {
        typename Tensor<n>::iterator ijk_b = jk_b;
        for (index i = a1; i--; ijk_a++, ijk_b += a2) {
          *ijk_b = *ijk_a;
        }
      }
    }
  }

  template<typename n>
  void permute_23(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3, index a4)
  {
    // In an abstract sense b(i,k,j,l) = a(i,j,k,l)
    // Both tensors are stored in row-major order. So for A we have
    // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
    //
    typename Tensor<n>::const_iterator ijkl_a = a.begin();
    typename Tensor<n>::iterator l_b = b.begin();
    index a12 = a1*a2;
    index a123 = a12*a3;
    for (; a4--; l_b += a123) {
      typename Tensor<n>::iterator kl_b = l_b;
      for (index k = a3; k--; kl_b += a1) {
        typename Tensor<n>::iterator kjl_b = kl_b;
        for (index j = a2; j--; kjl_b += a12) {
          typename Tensor<n>::iterator ikjl_b = kjl_b;
          for (index i = a1; i--; ijkl_a++, ikjl_b++) {
            *ikjl_b = *ijkl_a;
          }
        }
      }
    }
  }

  template<typename n>
  void permute_24(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3,
                  index a4, index a5)
  {
    // In an abstract sense b(i,l,k,j,m) = a(i,j,k,l,m)
    // Both tensors are stored in row-major order. So for A we have
    // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
    //
    typename Tensor<n>::const_iterator ijklm_a = a.begin();
    typename Tensor<n>::iterator m_b = b.begin();
    index a12 = a1*a2;
    index a123 = a12*a3;
    index a1234 = a123*a4;
    for (; a5--; m_b += a1234) {
      typename Tensor<n>::iterator lm_b = m_b;
      for (index l = a4; l--; lm_b += a1) {
        typename Tensor<n>::iterator lkm_b = lm_b;
        for (index k = a3; k--; lkm_b += a12) {
          typename Tensor<n>::iterator lkjm_b = lkm_b;
          for (index j = a2; j--; lkjm_b += a123) {
            typename Tensor<n>::iterator ilkjm_b = lkjm_b;
            for (index i = a1; i--; ijklm_a++, ilkjm_b++) {
              *ilkjm_b = *ijklm_a;
            }
          }
        }
      }
    }
  }

  template<typename n>
  const Tensor<n> do_permute(const Tensor<n> &a, index ndx1, index ndx2)
  {
    index n1 = normalize_index(ndx1, a.rank());
    index n2 = normalize_index(ndx2, a.rank());
    if (n2 < n1) {
      std::swap(n1,n2);
    } else if (n2 == n1) {
      return a;
    }
    Indices new_dims = a.dimensions();
    index i,a1,a2,a3,a4,a5;
    for (i = 0, a1 = 1; i < n1; )
      a1 *= new_dims[i++];
    a2 = new_dims[i++];
    for (a3 = 1; i < n2;)
      a3 *= new_dims[i++];
    a4 = new_dims[i++];
    for (a5 = 1; i < a.rank(); )
      a5 *= new_dims[i++];

    std::swap(new_dims.at(n1), new_dims.at(n2));
    Tensor<n> output(new_dims);

    if (a3 > 1) {
      permute_24(output, a, a1,a2,a3,a4,a5);
    } else if (a1 > 1) {
      permute_23(output, a, a1,a2,a4,a5);
    } else {
      permute_12(output, a, a2,a4,a5);
    }
    return output;
  }

} // namespace tensor
