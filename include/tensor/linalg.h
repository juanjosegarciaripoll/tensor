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

#ifndef TENSOR_LINALG_H
#define TENSOR_LINALG_H

#include <tensor/tensor.h>

/*!\addtogroup Linalg*/
/** Namespace for Linear Algebra functions based on BLAS, LAPACK, Lanczos and related algorithms. */
namespace linalg {

  using tensor::RTensor;
  using tensor::CTensor;

  const RTensor solve(const RTensor &A, const RTensor &B);
  const CTensor solve(const CTensor &A, const CTensor &B);

  extern bool accurate_svd;

  RTensor svd(RTensor A, RTensor *pU = 0, RTensor *pVT = 0, bool economic = 0);
  RTensor svd(CTensor A, CTensor *pU = 0, CTensor *pVT = 0, bool economic = 0);

  /**Eigenvalue decomposition of a real matrix.*/
  const CTensor eig(const RTensor &A, CTensor *R = 0, CTensor *L = 0);

  /**Eigenvalue decomposition of a complex matrix.*/
  const CTensor eig(const CTensor &A, CTensor *R = 0, CTensor *L = 0);

  double eig_power_right(const RTensor &A, RTensor *vector,
                         size_t iter = 0, double tol = 1e-11);
  double eig_power_left(const RTensor &A, RTensor *vector,
                        size_t iter = 0, double tol = 1e-11);
  tensor::cdouble eig_power_right(const CTensor &A, CTensor *vector,
                                  size_t iter = 0, double tol = 1e-11);
  tensor::cdouble eig_power_left(const CTensor &A, CTensor *vector,
                                 size_t iter = 0, double tol = 1e-11);

  RTensor eig_sym(const RTensor &A, RTensor *pR = 0);
  RTensor eig_sym(const CTensor &A, CTensor *pR = 0);

  const RTensor expm(const RTensor &A, unsigned int order = 7);
  const CTensor expm(const CTensor &A, unsigned int order = 7);

} // namespace linalg


#endif // !TENSOR_LINALG_H
