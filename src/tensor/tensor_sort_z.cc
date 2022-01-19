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
#include <functional>
#include <tensor/tensor.h>

namespace tensor {

bool operator<(const cdouble &a, const cdouble &b) {
  return (real(a) < real(b));
}

CTensor sort(const CTensor &v, bool reverse) {
  CTensor output(v);
  if (reverse) {
    std::sort(output.begin(), output.end(), std::greater<double>());
  } else {
    std::sort(output.begin(), output.end(), std::less<double>());
  }
  return output;
}

template <typename elt_t>
struct Compare {
  const elt_t *p;

  Compare(const elt_t *newp) : p(newp){};
  int operator()(size_t i1, size_t i2) { return p[i1] < p[i2]; }
};

template <typename elt_t>
struct CompareInv {
  const elt_t *p;

  CompareInv(const elt_t *newp) : p(newp){};
  int operator()(size_t i1, size_t i2) { return p[i1] > p[i2]; }
};

Indices sort_indices(const CTensor &v, bool reverse) {
  if (v.size()) {
    Indices output = iota(0, v.size() - 1);
    if (reverse) {
      CompareInv<double> c(v.begin());
      std::sort(output.begin(), output.end(), c);
    } else {
      Compare<double> c(v.begin());
      std::sort(output.begin(), output.end(), c);
    }
    return output;
  } else {
    return Indices();
  }
}

}  // namespace tensor
