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

#include <cassert>
#include <tensor/tensor.h>

namespace tensor {

  template<typename n> inline
  const Tensor<n> do_diag(const Tensor<n> &a, int which, int rows, int cols)
  {
    Tensor<n> output(rows, cols);
    output.fill_with_zeros();
    index r0, c0;
    if (which < 0) {
      r0 = -which;
      c0 = 0;
    } else {
      r0 = 0;
      c0 = which;
    }
    index l = std::min<index>(rows - r0, cols - c0);
    if (l < 0) {
      std::cerr << "In diag(a,which,...) the value of WHICH exceeds the size of the matrix"
                << std::endl;
      abort();
    }
    if (l != a.size()) {
      std::cerr << "In diag(a,...) the vector A has too few/many elements."
                << std::endl;
      abort();
    }
    for (size_t i = 0; i < (size_t)l; i++) {
	output.at(r0+i,c0+i) = a[i];
    }
    return output;
  }

} // namespace tensor
