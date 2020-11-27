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

#include <numeric>
#include <functional>
#include <tensor/io.h>
#include <tensor/indices.h>

namespace tensor {

const ListGenerator<index> igen = {};
const ListGenerator<double> rgen = {};
const ListGenerator<cdouble> cgen = {};

template class Vector<index>;

bool all_equal(const Indices &a, const Indices &b) {
  return (a.size() == b.size()) &&
         std::equal(a.begin_const(), a.end_const(), b.begin_const());
}

index Indices::total_size() const {
  if (size()) {
    index aux = 1;
    for (const_iterator it = begin_const(); it != end_const(); ++it) {
      if (*it < 0) {
        std::cerr << "Negative dimension in tensor's dimension #"
                  << (it - begin_const()) << std::endl
                  << "All dimensions:" << std::endl
                  << *this << std::endl;
        return false;
      }
      aux *= *it;
    }
    return aux;
  } else {
    return 0;
  }
}

const Indices Indices::range(index min, index max, index step) {
  if (max < min) {
    return Indices();
  } else {
    index size = (max - min) / step + 1;
    Indices output(size);
    for (Indices::iterator it = output.begin(); it != output.end(); it++) {
      *it = min;
      min += step;
    }
    return output;
  }
}

}  // namespace tensor
