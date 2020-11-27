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

namespace tensor {

template <typename n>
static void trace_loop(n *C, const n *D, index a1, index a2, index a3, index a4,
                       index a5) {
  // C(a1,a3,a5), D(a1,a2,a3,a4,a5)
  // C(i,k,m) = sum D(i,l,k,l,m) over l
  index a2b = std::min(a2, a4);
  if (a1 == 1) {
    for (index m = 0; m < a5; m++) {
      for (index l = 0; l < a2b; l++) {
        for (index k = 0; k < a3; k++) {
          C[k + a3 * m] += D[l + a2 * (k + a3 * (l + a4 * m))];
        }
      }
    }
  } else {
    for (index m = 0; m < a5; m++) {
      for (index l = 0; l < a2b; l++) {
        for (index k = 0; k < a3; k++) {
          for (index i = 0; i < a1; i++) {
            C[i + a1 * (k + a3 * m)] +=
                D[i + a1 * (l + a2 * (k + a3 * (l + a4 * m)))];
          }
        }
      }
    }
  }
}

template <typename elt_t>
static const Tensor<elt_t> do_trace(const Tensor<elt_t> &D, index i1,
                                    index i2) {
  assert(i1 < D.rank() && i1 > -D.rank());
  assert(i2 < D.rank() && i2 > -D.rank());
  if (i1 < 0) i1 = i1 + D.rank();
  if (i2 < 0) i2 = i2 + D.rank();
  if (i1 > i2) {
    std::swap(i1, i2);
  } else if (i2 == i1) {
    std::cerr << "In trace(D, i, j), indices 'i' and 'j' are the same."
              << std::endl;
    abort();
  }

  index a1, a2, a3, a4, a5, i, rank;
  Indices dimensions(std::max(D.rank() - 2, 1));
  dimensions.at(rank = 0) = 1;
  for (a1 = 1, i = 0; i < i1;) {
    a1 *= (dimensions.at(rank++) = D.dimension(i++));
  }
  a2 = D.dimension(i++);
  for (a3 = 1; i < i2;) a3 *= (dimensions.at(rank++) = D.dimension(i++));
  a4 = D.dimension(i++);
  for (a5 = 1; i < D.rank();) a5 *= (dimensions.at(rank++) = D.dimension(i++));

  Tensor<elt_t> output = RTensor::zeros(dimensions);
  trace_loop<elt_t>(output.begin(), D.begin(), a1, a2, a3, a4, a5);
  return output;
}

}  // namespace tensor
