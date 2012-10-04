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

#include <tensor/linalg.h>
#include <mps/quantum.h>

namespace mps {

  /**Compute the von Neumann entropy. If the input tensor is a vector,
     it assumes it is a list of eigenvalues from a density matrix and
     computes the associated entropy, \f$-\sum_i \lambda_l
     \log(\lambda_i)\f$. If the input state is a matrix, this matrix
     is diagonalized and the eigenvalues are used to compute the
     entropy. */
  double entropy(const RTensor &t)
  {
    if (t.rank() == 1) {
      const RTensor &l = t;
      double ltot = abs(sum(l)), s = 0.0;
      for (size_t i = 0; i < l.size(); i++) {
	double li = abs(l[i]) / ltot;
	s -= li * log(li);
      }
      return s;
    } else if (t.rank() == 2) {
      return entropy(linalg::eig_sym(t));
    } else {
      assert(t.rank() == 1 || t.rank() == 2);
      abort();
    }
  }

}
