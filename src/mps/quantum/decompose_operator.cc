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
#include <mps/tools.h>

namespace mps {

  //----------------------------------------------------------------------
  // SCHMIDT DECOMPOSITION FOR OPERATORS
  //
  // U is an operator that acts on two spins of dimension "d". We decompose
  // it as a sum of local operators
  //	U = sum(k, O1k, O2k)
  // More precisely,
  //	U([j1,j2],[i1,i2]) = sum(d,O1(i1,j1,d),O2(i2,j2,d));
  //

  template <class tensor>
  static inline void do_decompose_operator(const tensor &U0, tensor *O1, tensor *O2)
  {
    /*
     * Notice the funny reordering of indices in O1 and O2, which is due to the
     * following statement and which simplifies the application of O1 and O2 on a
     * given vector.
     */
    index d1 = (index)sqrt((double)U0.rows());
    index d2 = d1;

    if (U0.rows() != U0.columns()) {
      std::cerr << "The routine decompose_operator() can only act on square matrices and you\n"
		<< "have passed a matrix that is " << U0.rows() << " by " << U0.columns();
      abort();
    }

    tensor U = reshape(permute(reshape(U0, d1,d2,d1,d2), 1,2), d1*d1,d2*d2);
#if 1
    RTensor s = sqrt(limited_svd(U, O1, O2));
    scale_inplace(*O1, -1, s);
    scale_inplace(*O2, 0, s);
#else
    RTensor s = sqrt(svd(U, O1, O2, 1));
    scale_inplace(*O1, -1, s);
    scale_inplace(*O2, 0, s);
#endif
    *O1 = reshape(*O1, d1,d1,s.size());
    *O2 = reshape(transpose(*O2), d2,d2,s.size());
  }

} // namespace mps
