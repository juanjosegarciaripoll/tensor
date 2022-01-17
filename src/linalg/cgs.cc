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

#include <functional>
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace linalg {

using namespace tensor;

template <class Tensor>
static Tensor solve_cgs(const LinearMap<Tensor> &A, const Tensor &b,
                        const Tensor *x_start, tensor::index maxiter,
                        double tol) {
  typedef typename Tensor::elt_t number;
  if (maxiter == 0) {
    maxiter = b.rows();
  }
  if (tol <= 0) {
    tol = 1e-10;
  }
  Tensor x = x_start ? *x_start : (b + 0.05 * Tensor::random(b.dimensions()));
  Tensor r = b - A(x);
  Tensor p = r;
  number rsold = scprod(r, r);
  if (sqrt(abs(rsold)) > tol) {
    while (maxiter-- >= 0) {
      Tensor Ap = A(p);
      number beta = scprod(p, Ap);
      if (abs(beta) < 1e-15 * abs(rsold)) {
        // We have hit a zero
        std::cerr << "Singular system of equations hit in cgs()" << std::endl;
        abort();
      }
      number alpha = rsold / scprod(p, Ap);
      x += alpha * p;
      r -= alpha * Ap;
      number rsnew = scprod(r, r);
      if (sqrt(abs(rsnew)) < tol) break;
      p = r + (rsnew / rsold) * p;
      rsold = rsnew;
    }
  }
  return x;
}

}  // namespace linalg
