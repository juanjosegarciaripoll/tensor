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

  const RMPS product_state(index length, const RTensor &local_state)
  {
    assert(local_state.rank() == 1);
    assert(local_state.size());
    assert(length > 0);

    index d = local_state.size();
    RMPS output(length);
    std::fill(output.begin(), output.end(), reshape(local_state, 1, d, 1));
    return output;
  }

} // namespace mps
