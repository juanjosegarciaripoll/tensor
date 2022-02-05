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

#include <tensor/io.h>

namespace tensor {

template <typename elt_t>
std::ostream &MatrixForm<elt_t>::display(std::ostream &s) const {
  if (data.rank() > 2) {
    std::cerr << "MatrixForm can only be used with two-dimensional tensors.\n";
    abort();
  } else if (data.rank() == 2) {
    index rows = data.rows();
    index cols = data.columns();
    for (index i = 0; i < rows; i++) {
      if (i == 0)
        s << '[';
      else
        s << "\n ";
      for (index j = 0; j < cols; j++) {
        if (j) s << ", ";
        s << data(i, j);
      }
    }
    s << ']';
  } else if (data.rank() == 1) {
    for (index i = 0; i < data.ssize(); i++) {
      if (i)
        s << ", ";
      else
        s << '[';
      s << data[i];
    }
    s << ']';
  }
  return s;
}

}  // namespace tensor
