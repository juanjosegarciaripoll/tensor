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

#include <tensor/sparse.h>
#include "loops.h"

namespace tensor_test {

template <typename elt_t>
class kron_2d_fixture : public std::vector<Tensor<elt_t> > {
 public:
  kron_2d_fixture() {
#define add(a, b) this->push_back(Tensor<elt_t>(a, b))
    add(0, 0);
    add(1, 0);
    add(0, 0);

    add(igen << 1 << 1, gen<elt_t>(1));
    add(igen << 1 << 1, gen<elt_t>(-2));
    add(igen << 1 << 1, gen<elt_t>(-2));

    add(igen << 1 << 2, gen<elt_t>(1) << 2);
    add(igen << 2 << 3, gen<elt_t>(1) << 7 << 3 << 9 << 5 << 11);
    add(igen << 2 << 6, gen<elt_t>(1) << 7 << 3 << 9 << 5 << 11 << 2 << 14 << 6
                                      << 18 << 10 << 22);

    add(igen << 2 << 1, gen<elt_t>(1) << 2);
    add(igen << 2 << 3, gen<elt_t>(1) << 7 << 3 << 9 << 5 << 11);
    add(igen << 4 << 3, gen<elt_t>(1) << 7 << 2 << 14 << 3 << 9 << 6 << 18 << 5
                                      << 11 << 10 << 22);
#undef add
  }
};

}  // namespace tensor_test
