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
#include <mps/itebd.h>

namespace mps {

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(tensor::index dimension) : 
    A_(igen << 1 << dimension << 1),
    B_(igen << 1 << dimension << 1),
    lA_(igen << 1, gen<elt_t>(1)),
    lB_(igen << 1, gen<elt_t>(1))
  {
    assert(dimension > 0);
  }
    
  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &A)
  : A_(reshape(A / norm2(A), igen << 1 << A.dimension(0) << 1)),
    B_(reshape(A / norm2(A), igen << 1 << A.dimension(0) << 1)),
    lA_(igen << 1, gen<elt_t>(1.0)),
    lB_(igen << 1, gen<elt_t>(1.0)),
    AlA_(A_), BlB_(B_), canonical_(1)
  {
    assert(A.rank() == 1);
  }

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &A, const Tensor &B) :
    A_(reshape(A / norm2(A), igen << 1 << A.dimension(0) << 1)),
    B_(reshape(B / norm2(B), igen << 1 << A.dimension(0) << 1)),
    lA_(igen << 1, gen<elt_t>(1.0)),
    lB_(igen << 1, gen<elt_t>(1.0)),
    AlA_(A_), BlB_(B_), canonical_(1)
  {
    assert(A.rank() == 1);
    assert(A.rank() == 1);
  }

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &A, const Tensor &lA,
		       const Tensor &B, const Tensor &lB,
                       bool canonical) :
    A_(A), lA_(lA), B_(B), lB_(lB),
    AlA_(scale(A, -1, lA)), BlB_(scale(B, -1, lB)),
    canonical_(canonical)
  {
    assert(A_.rank() == 3);
    assert(A_.dimension(0) == lB.dimension(0));
    assert(A_.dimension(2) == lA.dimension(0));
    assert(B_.dimension(0) == lA.dimension(0));
    assert(B_.dimension(2) == lB.dimension(0));
  }

  template<class Tensor>
  typename Tensor::elt_t
  iTEBD<Tensor>::expected_value(const Tensor &Op, int site) const
  {
    assert(is_canonical());
    Tensor v = left_boundary(site);
    elt_t value = trace(propagate_right(v, combined_matrix(site), Op));
    elt_t norm = trace(propagate_right(v, combined_matrix(site), Op));
    return value / real(norm);
  }

  template<class Tensor>
  typename Tensor::elt_t
  iTEBD<Tensor>::expected_value(const Tensor &Op1, const Tensor &Op2) const
  {
    return string_order(Op1, 0, Tensor(), Op2, 1);
  }

  template<class Tensor>
  typename Tensor::elt_t
  iTEBD<Tensor>::expected_value(const Tensor &Op1, int i, const Tensor &Op2, int j) const
  {
    return string_order(Op1, i, Tensor(), Op2, j);
  }

  template<class Tensor>
  typename Tensor::elt_t
  iTEBD<Tensor>::string_order(const Tensor &Opi, int i, const Tensor &Opmiddle,
                              const Tensor &Opj, int j) const
  {
    if (i > j)
      return string_order(Opj, j, Opmiddle, Opi, i);
    assert(is_canonical());
    int site = i;
    Tensor v1 = left_boundary(site);
    Tensor v2 = v1;
    v1 = propagate_right(v1, combined_matrix(site), Opi);
    v2 = propagate_right(v2, combined_matrix(site));
    ++site;
    while (site < j) {
      v1 = propagate_right(v1, combined_matrix(site), Opmiddle);
      v2 = propagate_right(v2, combined_matrix(site));
      ++site;
    }
    elt_t value = trace(propagate_right(v1, combined_matrix(site), Opj));
    elt_t norm = trace(propagate_right(v2, combined_matrix(site))); 
    return value / real(norm);
  }

  template<class Tensor>
  typename Tensor::elt_t
  iTEBD<Tensor>::expected_value12(const Tensor &Op12, int site) const
  {
    tensor::index a, i, b, j, c;
    const Tensor &AlA = (site & 1) ? BlB_ : AlA_;
    const Tensor &B = (site & 1) ? A_ : B_;
    const Tensor &lB = (site & 1) ? lA_ : lB_;
    Tensor AlAB = reshape(fold(AlA, -1, B, 0), a, i*j, c);
    Tensor v = left_boundary(site);
    elt_t value = trace(propagate_right(v, AlAB, Op12));
    elt_t norm = trace(propagate_right(v, AlAB));
    return value / real(norm);
  }

  template<class Tensor>
  double
  iTEBD<Tensor>::energy(const Tensor &Op12) const
  {
    return real(expected_value12(Op12, 0) + expected_value12(Op12, 1));
  }

  template<class Tensor>
  double iTEBD<Tensor>::entropy(int site) const
  {
    Tensor l = left_boundary(site);
    double ltot = abs(sum(l)), s = 0.0;
    for (size_t i = 0; i < l.size(); i++) {
      double li = abs(l[i]) / ltot;
      s -= li * log(li);
    }
    return s;
  }

}
