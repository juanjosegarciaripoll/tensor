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

#include <float.h>
#include <mps/tools.h>

namespace mps {

size_t
where_to_truncate(const RTensor &s, double tol, tensor::index max_a2)
{
  /* S is a vector of positive numbers arranged in decreasing order.  This
   * routine looks for a point to truncate S such that the norm-2 error made
   * is smaller than the relative tolerance (TOL) or the length of the output
   * is smaller than MAX_A2.
   */
  size_t L = s.size();
  if (max_a2 == 0 || max_a2 > L) {
    max_a2 = L;
  }
  if (tol == 0) {
    /* If the tolerance is zero, we only drop the trailing zero elements. There
     * is no need to accumulate values. */
    for (size_t i = L; i--; ) {
      if (s[i]) {
        return (i < max_a2)? (i+1) : max_a2;
      }
    }
    return 0;
  }
  /*
   * cumulated[i] contains the norm of the elements _beyond_ the i-th
   * site. This means that if we keep (i+1) leading elements, the error will
   * be exactly cumulated[i].
   */
  double *cumulated = new double[L];
  double total = 0;
  for (size_t i = L; i--; ) {
    cumulated[i] = total;
    total += square(s[i]);
  }
  /* Due to the precision limits in current processors, we automatically
   * relax the tolerance to DBL_EPSILON, which is a floating point number
   * such that added to 1.0 gives 1.0. In other words, a tolerance <=
   * DBL_EPSILON is irrelevant for all purposes.
   */
  if (tol < 0) {
    tol = DBL_EPSILON;
  }
  double limit = tol * total;
  for (size_t i = 0; i < max_a2; i++) {
    if (cumulated[i] <= limit) {
      max_a2 = i+1;
      break;
    }
  }
  delete[] cumulated;
  return max_a2;
}

}

