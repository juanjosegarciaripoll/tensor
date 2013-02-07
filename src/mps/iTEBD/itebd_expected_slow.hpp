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

#include <mps/itebd.h>
#include <mps/tools.h>

namespace mps {

  template<class t>
  typename t::elt_t slow_expected(const t &Op1, t A, const t &lA)
  {
    std::cout << "A=" << A << std::endl;
    A = scale(A, -1, lA);
    std::cout << "A=" << A << std::endl;
    t R0 = build_E_matrix(A);
    t R = R0;
    for (size_t i = 1; i <= 10; i++) {
      R = mmult(R, R);
      R = R / norm2(R);
    }
    t R2 = build_E_matrix(foldin(Op1, -1, A, 1), A);
    typename t::elt_t N = trace(mmult(R0, R));
    typename t::elt_t E = trace(mmult(R2, R));
    return (E / N);
  }

  template<class t>
  typename t::elt_t slow_expected12(const t &Op1, const t &Op2, t A, const t &lA)
  {
    A = scale(A, -1, lA);
    t R0 = build_E_matrix(A);
    t R = R0;
    for (size_t i = 1; i <= 10; i++) {
      R = mmult(R, R);
      R = R / norm2(R);
    }
    t R1 = build_E_matrix(foldin(Op1, -1, A, 1), A);
    t R2 = build_E_matrix(foldin(Op2, -1, A, 1), A);
    typename t::elt_t N = trace(mmult(R0, mmult(R0, R)));
    typename t::elt_t E = trace(mmult(R1, mmult(R2, R)));
    return (E / N);
  }

  template<class t>
  const t ensure_3_indices(const t &A)
  {
    tensor::index l = A.size();
    tensor::index a = A.dimension(0);
    tensor::index b = A.dimension(A.rank()-1);
    return reshape(A, a, l/(a*b), b);
  }

  template<class t>
  typename t::elt_t slow_expected12(const t &Op12, const t &A, const t &lA, const t &B, const t &lB)
  {
    return slow_expected(Op12, ensure_3_indices(fold(scale(A, -1, lA), -1, B, 0)), lB);
  }

  template<class t>
  typename t::elt_t slow_expected12(const t &Op1, const t &Op2, const t &A, const RTensor &lA, const t &B, const RTensor &lB)
  {
    return slow_expected12(kron2(Op1, Op2), A, lA, B, lB);
  }

  template<class t>
  typename t::elt_t slow_expected1(const t &Op1, const t &A, const t &lA, const t &B, const t &lB)
  {
    t id = t::eye(B.get_dim(1));
    return slow_expected12(Op1, id, A, lA, B, lB);
  }

  template<class t>
  typename t::elt_t slow_expected2(const t &Op2, const t &A, const t &lA, const t &B, const t &lB)
  {
    t id = t::eye(A.get_dim(1));
    return slow_expected12(id, Op2, A, lA, B, lB);
  }

  template<class t>
  typename t::elt_t slow_expected12(const iTEBD<t> &psi, const t &Op12)
  {
    slow_expected12(Op12, psi.matrix(0), psi.right_vector(0),
		    psi.matrix(1), psi.right_vector(0));
  }

} // namespace mps
