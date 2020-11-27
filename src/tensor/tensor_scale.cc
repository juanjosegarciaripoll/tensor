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

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// SCALE A TENSOR
//
// 1) IN PLACE
//

template <typename t1, typename t2>
void doscale(t1 *p1, const t2 *p20, index d1, index d2, index d3) {
  index i, j;
  const t2 *p2;
  if (d1 == 1) {
    for (; d3; d3--) {
      for (i = d2, p2 = p20; i; i--, p1++, p2++) {
        *p1 *= *p2;
      }
    }
  } else {
    for (; d3; d3--) {
      for (i = d2, p2 = p20; i; i--, p2++) {
        t2 r = *p2;
        for (j = d1; j; j--, p1++) {
          *p1 *= r;
        }
      }
    }
  }
}

template <typename elt_t>
void scale_inplace(Tensor<elt_t> &t, int ndx, const Vector<double> &v) {
  index d1, d2, d3;
  surrounding_dimensions(t.get_dims(), t.normal_index(ndx), &d1, &d2, &d3);
  if (d2 != v.size()) {
    std::cerr << "In scale() the dimension " << ndx
              << " of the tensor does not match the length " << v.size()
              << " of the scale vector" << std::endl;
    abort();
  }
  doscale(t.begin(), v.begin_const(), d1, d2, d3);
}

//
// 2) OUT OF PLACE
//

template <class t1, class t2, class t3>
void doscale(t1 *p1, const t2 *p2, const t3 *p30, index d1, index d2,
             index d3) {
  index i, j;
  const t3 *p3;
  if (d1 == 1) {
    for (; d3; d3--) {
      for (i = d2, p3 = p30; i; i--, p1++, p2++, p3++) {
        *p1 = *p2 * *p3;
      }
    }
  } else {
    for (; d3; d3--) {
      for (i = d2, p3 = p30; i; i--, p3++) {
        const t3 r = *p3;
        for (j = d1; j; j--, p1++, p2++) {
          *p1 = r * *p2;
        }
      }
    }
  }
}

}  // namespace tensor
