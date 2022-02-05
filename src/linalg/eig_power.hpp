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
#include <tensor/io.h>

namespace linalg {

using namespace tensor;

template <typename elt_t>
elt_t eig_power_loop(const LinearMap<Tensor<elt_t>> &A, size_t dims,
                     Tensor<elt_t> *vector, size_t iter, double tol) {
  if (tol <= 0) {
    tol = 1e-11;
  }
  assert(vector);
  Tensor<elt_t> &v = *vector;
  if (v.size() != dims) {
    v = 0.5 - Tensor<elt_t>::random(static_cast<tensor::index>(dims));
  }
  if (iter == 0) {
    iter = std::max<size_t>(20, v.size());
  }
  v /= norm2(v);
  elt_t eig = 0;
  //
  // We apply repeatedly the map 'A' onto the same random initial
  // vector, until (A^n)*v converges to the eigenstate with the largest
  // eigenvalue (in absolute value) that has some support on 'v'.
  //
  for (size_t i = 0; i <= iter; i++) {
    Tensor<elt_t> v_new = A(v);
    eig = scprod(v, v_new);
    double err = norm0(v_new - eig * v);
    // Stop if the vector is sufficiently close to an eigenstate
    if (err < tol * std::abs(eig)) break;
    v = (v_new /= norm2(v_new));
  }
  return eig;
}

}  // namespace linalg
