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
#include <tensor/tensor.h>

namespace tensor {

Indices squeeze_dimensions(const Indices &d) {
  auto unused = std::count(d.begin(), d.end(), 1);
  Indices output(d.size() - static_cast<size_t>(unused));
  Indices::const_iterator b = d.begin();
  for (Indices::iterator a = output.begin(); a != output.end() && b != d.end();
       b++) {
    if (*b > 1) *(a++) = *b;
  }
  return output;
}

}  // namespace tensor
