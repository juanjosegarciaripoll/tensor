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
#include <tensor/tensor_blas.h>

namespace tensor {

  Tensor<double> &operator+=(Tensor<double> &a, const Tensor<double> &b) {
    assert(a.size() == b.size());
#if 1
    Tensor<double>::iterator ita = a.begin();
    Tensor<double>::iterator itae = a.end();
    Tensor<double>::const_iterator itb = b.begin();
    while (ita != itae) {
      (*ita) += (*itb);
      ++ita;
      ++itb;
    }
#else
    cblas_daxpy(a.size(),
		1.0, static_cast<const double*>((void*)b.begin_const()), 1,
                static_cast<double*>((void*)a.begin()), 1);
#endif
    return a;
  }

} // namespace tensor
