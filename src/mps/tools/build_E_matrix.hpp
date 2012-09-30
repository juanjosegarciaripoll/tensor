// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <mps/tools.h>

namespace mps {

  using namespace tensor;

  template<typename elt_t>
  static inline const Tensor<elt_t>
  do_build_E_matrix(const Tensor<elt_t> &A1, const Tensor<elt_t> &A2, tensor::index *l, tensor::index *r)
  {
    Tensor<elt_t> R = fold(A1, 1, conj(A2), 1);
    tensor::index a, b;
    R.get_dimensions(&a,&b,&a,&b);
    tensor::index a2 = a*a, b2 = b*b;
    R = reshape(permute(R, 1, 2), a2, b2);
    if (l) *l = a;
    if (r) *r = b;
    return R;
  }

}
