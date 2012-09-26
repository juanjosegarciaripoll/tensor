// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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

#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace tensor;

  template<typename elt_t>
  elt_t eig_power_loop(const Tensor<elt_t> &O, Tensor<elt_t> *vector,
                       bool right, size_t iter)
  {
    if (O.columns() != O.rows()) {
      std::cerr << "eig_right_power cannot solve non-square problems";
      abort();
    }
    if (iter == 0) {
      iter = 20;
    }
    Tensor<elt_t> v(O.columns());
    v.randomize();
    v = v / norm2(v);
    elt_t old_eig;
    for (size_t i = 0; i <= iter; i++) {
      Tensor<elt_t> v_new = fold(O, right? -1 : 0, v, 0);
      double n = norm2(v_new);
      v = v_new / n;
      elt_t eig = scprod(v_new, v);
      if (i) {
        double confidence = abs(eig - old_eig);
        if (confidence < 1e-11 * abs(eig)) {
          break;
        }
      }
      old_eig = eig;
    }
    *vector = v;
    return old_eig;
  }

} // namespace linalg
