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

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_KRON_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_KRON_HPP

#include <cassert>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// KRONECKER PRODUCT OF TENSORS
//

template <typename elt_t>
static inline Tensor<elt_t> do_kron(const Tensor<elt_t> &b,
                                    const Tensor<elt_t> &a) {
  assert(b.rank() == a.rank());
  assert(b.rank() <= 2);
  if (a.rank() == 1) {
    // FIXME! CHAPUZA!
    return reshape(kron(reshape(b, b.ssize(), 1), reshape(a, a.ssize(), 1)),
                   a.ssize() * b.ssize());
  }

  // C([i,j],[k,l]) = A(i,k) B(j,l)
  const index i_len = a.rows();
  const index j_len = b.rows();
  const index k_len = a.columns();
  const index l_len = b.columns();
  const index ij_len = i_len * j_len;
  const index kl_len = k_len * l_len;
  if (ij_len == 0 || kl_len == 0) return Tensor<elt_t>::empty(ij_len, kl_len);

  auto output = Tensor<elt_t>::empty(ij_len, k_len * l_len);
  typename Tensor<elt_t>::iterator pc = output.begin();
  typename Tensor<elt_t>::const_iterator pb = b.begin();
  for (index l = 0; l < l_len; l++, pb += j_len) {
    typename Tensor<elt_t>::const_iterator pa = a.begin();
    for (index k = 0; k < k_len; k++, pa += i_len) {
#ifdef MPS_USE_CBLAS
      memset(pc, 0, ij_len * sizeof(*pc));
      geru(i_len, j_len, Tensor<elt_t>::elt_one(), pa, 1, pb, 1, pc, i_len);
      pc += i_len * j_len;
#else
      typename Tensor<elt_t>::const_iterator pb2 = pb;
      for (index j = 0; j < j_len; j++, pb2++) {
        typename Tensor<elt_t>::const_iterator pa2 = pa;
        const elt_t v = *pb2;
        for (index i = 0; i < i_len; i++, pc++, pa2++) {
          *pc = v * (*pa2);
        }
      }
#endif
    }
  }
  return output;
}

template <typename elt_t>
const Tensor<elt_t> do_kron2_sum(const Tensor<elt_t> &a,
                                 const Tensor<elt_t> &b) {
  assert(a.rank() == b.rank());
  assert(a.rank() <= 2);
  if (a.rank() == 1) {
    // FIXME! CHAPUZA!
    index a1 = a.ssize();
    index b1 = b.ssize();
    return reshape(kron2(reshape(a, a1, 1), Tensor<elt_t>::ones(b1, 1)) +
                       kron2(Tensor<elt_t>::ones(a1, 1), reshape(b, b1, 1)),
                   a1 * b1);
  } else {
    index a1, a2, b1, b2;
    a.get_dimensions(&a1, &a2);
    b.get_dimensions(&b1, &b2);
    return kron(Tensor<elt_t>::eye(b1, b2), a) +
           kron(b, Tensor<elt_t>::eye(a1, a2));
  }
}

}  // namespace tensor

#endif  // TENSOR_DETAIL_TENSOR_KRON_HPP
