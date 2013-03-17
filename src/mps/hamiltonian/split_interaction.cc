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

#include <cmath>
#include <tensor/linalg.h>
#include <mps/quantum.h>
#include <mps/hamiltonian.h>

namespace mps {

  using namespace tensor;

  void
  split_interaction(const CTensor &H12, std::vector<CTensor> *v1, std::vector<CTensor> *v2)
  {
    assert(H12.rank() == 2);

    CTensor O1, O2;
    RTensor s = linalg::svd(H12, &O1, &O2, SVD_ECONOMIC);

    index N = sqrt(H12.dimension(0));
    index n_op = s.size();
    v1->resize(n_op);
    v2->resize(n_op);
    for (index i = 0; i < N; i++) {
      double sqrts = sqrt(s[i]);
      v1->at(i) = sqrts * tensor::reshape(O1(range(), range(i)), N, N);
      v2->at(i) = conj(sqrts * tensor::reshape(O2(range(), range(i)), N, N));
    }
  }

} // namespace mps
