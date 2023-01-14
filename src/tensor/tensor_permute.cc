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

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

template <typename n>
void permute_12(Tensor<n> &b, const Tensor<n> &a, index a1, index a2,
                index a3) {
  // In an abstract sense b(j,i,k) = a(i,j,k)
  // Both tensors are stored in row-major order. So for A we have
  // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
  //
  typename Tensor<n>::const_iterator ijk_a = a.begin();
  typename Tensor<n>::iterator k_b = b.begin();
  for (; a3; --a3, k_b += (a1 * a2)) {
    typename Tensor<n>::iterator jk_b = k_b;
    for (index j = a2; j--; ++jk_b) {
      typename Tensor<n>::iterator ijk_b = jk_b;
      for (index i = a1; i--; ++ijk_a, ijk_b += a2) {
        //tensor_assert(ijk_a >= a.begin() && ijk_a < a.end());
        //tensor_assert(ijk_b >= b.begin() && ijk_b < b.end());
        *ijk_b = *ijk_a;
      }
    }
  }
}

template <typename n>
void permute_23(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3,
                index a4) {
  // In an abstract sense
  // b(i,k,j,l) = a(i,j,k,l)
  // b(a1,a3,a2,a4) = a(a1,a2,a3,a4)
  // Both tensors are stored in row-major order. So for A we have
  // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
  //
  typename Tensor<n>::const_iterator ijkl_a = a.begin();
  typename Tensor<n>::iterator l_b = b.begin();
  index a13 = a1 * a3;
  index a132 = a13 * a2;
  for (; a4--; l_b += a132) {
    typename Tensor<n>::iterator kl_b = l_b;
    for (index k = a3; k--; kl_b += a1) {
      typename Tensor<n>::iterator kjl_b = kl_b;
      for (index j = a2; j--; kjl_b += a13) {
        typename Tensor<n>::iterator ikjl_b = kjl_b;
        for (index i = a1; i--; ++ijkl_a, ++ikjl_b) {
          //tensor_assert(ijkl_a >= a.begin() && ijkl_a < a.end());
          //tensor_assert(ikjl_b >= b.begin() && ikjl_b < b.end());
          *ikjl_b = *ijkl_a;
        }
      }
    }
  }
}

template <typename n>
void permute_24(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3,
                index a4, index a5) {
  // In an abstract sense
  // b(i,l,k,j,m) = a(i,j,k,l,m)
  // b(a1,a4,a3,a2,a5) = a(a1,a2,a3,a4,a5)
  // Both tensors are stored in row-major order. So for A we have
  // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
  //
  typename Tensor<n>::const_iterator ijklm_a = a.begin();
  typename Tensor<n>::iterator m_b = b.begin();
  index a14 = a1 * a4;
  index a143 = a14 * a3;
  index a1432 = a143 * a2;
  for (; a5--; m_b += a1432) {
    typename Tensor<n>::iterator lm_b = m_b;
    for (index l = a4; l--; lm_b += a1) {
      typename Tensor<n>::iterator lkm_b = lm_b;
      for (index k = a3; k--; lkm_b += a14) {
        typename Tensor<n>::iterator lkjm_b = lkm_b;
        for (index j = a2; j--; lkjm_b += a143) {
          typename Tensor<n>::iterator ilkjm_b = lkjm_b;
          for (index i = a1; i--; ++ijklm_a, ++ilkjm_b) {
            //tensor_assert(ilkjm_b < b.begin() || b.end() <= ilkjm_b);
            //tensor_assert(ijklm_a < a.begin() || a.end() <= ijklm_a);
            *ilkjm_b = *ijklm_a;
          }
        }
      }
    }
  }
}

template <typename n>
void permute_13(Tensor<n> &b, const Tensor<n> &a, index a1, index a2, index a3,
                index a4) {
  // In an abstract sense
  // b(k,j,i,l) = a(i,j,k,l)
  // b(a3,a2,a1,a4) = a(a1,a2,a3,a4)
  // Both tensors are stored in row-major order. So for A we have
  // (0,0,0), (1,0,0), ... (2,1,0), ... (a1-1,a2-1,a3-1)
  //
  typename Tensor<n>::const_iterator ijkl_a = a.begin();
  typename Tensor<n>::iterator l_b = b.begin();
  index a32 = a3 * a2;
  index a321 = a32 * a1;
  for (; a4--; l_b += a321) {
    typename Tensor<n>::iterator kl_b = l_b;
    for (index k = a3; k--; ++kl_b) {
      typename Tensor<n>::iterator kjl_b = kl_b;
      for (index j = a2; j--; kjl_b += a3) {
        typename Tensor<n>::iterator kjil_b = kjl_b;
        for (index i = a1; i--; ++ijkl_a, kjil_b += a32) {
          //tensor_assert(kjil_b < b.begin() || b.end() <= kjil_b);
          //tensor_assert(ijkl_a < a.begin() || a.end() <= ijkl_a);
          *kjil_b = *ijkl_a;
        }
      }
    }
  }
}

template <typename n>
Tensor<n> do_permute(const Tensor<n> &a, index ndx1, index ndx2) {
  index n1 = Dimensions::normalize_index(ndx1, a.rank());
  index n2 = Dimensions::normalize_index(ndx2, a.rank());
  if (n2 < n1) {
    std::swap(n1, n2);
  } else if (n2 == n1) {
    return a;
  }
  Indices new_dims = a.dimensions();
  index i{0}, a1{1};
  for (i = 0, a1 = 1; i < n1;) a1 *= new_dims[i++];
  index a2 = new_dims[i++], a3{1};
  for (; i < n2;) a3 *= new_dims[i++];
  index a4 = new_dims[i++], a5{1};
  for (; i < a.rank();) a5 *= new_dims[i++];

  std::swap(new_dims.at(n1), new_dims.at(n2));

  if (a.size()) {
    if (a2 == 1) {
      a2 = a3;
      goto NO_A3;
    } else if (a4 == 1) {
      a4 = a3;
      goto NO_A3;
    }
    if (a3 > 1) {
      auto output = Tensor<n>::empty(new_dims);
      if (a1 > 1) {
        permute_24(output, a, a1, a2, a3, a4, a5);
      } else {
        permute_13(output, a, a2, a3, a4, a5);
      }
      return output;
    } else {
    NO_A3:
      if (a4 > 1 || a2 > 1) {
        auto output = Tensor<n>::empty(new_dims);
        if (a1 > 1) {
          permute_23(output, a, a1, a2, a4, a5);
        } else {
          permute_12(output, a, a2, a4, a5);
        }
        return output;
      }
    }
  }
  return reshape(a, new_dims);
}

}  // namespace tensor
