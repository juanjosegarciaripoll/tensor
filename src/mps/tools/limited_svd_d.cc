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
#include <mps/tools.h>

namespace mps {

  const RTensor
  limited_svd(RTensor A, RTensor *U, RTensor *V, double tolerance, tensor::index max_dim)
  {
    RTensor s = linalg::svd(A, U, V, true /* economic */);
    tensor::index c = where_to_truncate(s, tolerance, max_dim);
    *U = (*U)(range(), range(0,c-1));
    *V = (*V)(range(0,c-1), range());
    s = s(range(0,c-1));
    return s / norm2(s);
  }

}
