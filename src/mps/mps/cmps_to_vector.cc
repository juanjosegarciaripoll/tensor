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

#include <algorithm>
#include <mps/mps.h>

namespace mps {

  using namespace tensor;

  const CTensor mps_to_vector(const CMPS &mps)
  {
    assert(mps.size() > 0);

    CTensor output = mps[0];
    index d0 = output.dimension(0);
    index d = output.dimension(1);
    for (index i = 1; i < mps.size(); i++) {
      output = fold(output, -1, mps[i], 0);
      output = reshape(output, d0, output.dimension(1)*output.dimension(2), output.dimension(3));
    }
    return trace(output, 0, -1);
  }

} // namespace mps
