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
    this->push_back(RTensor::empty(0, 0));
    this->push_back(RTensor::empty(1, 0));
    this->push_back(RTensor::empty(0, 0));

    this->emplace_back(Dimensions{1, 1}, gen<elt_t>(1));
    this->emplace_back(Dimensions{1, 1}, gen<elt_t>(-2));
    this->emplace_back(Dimensions{1, 1}, gen<elt_t>(-2));

    this->emplace_back(Dimensions{1, 2}, gen<elt_t>(1) << 2);
    this->emplace_back(Dimensions{2, 3}, gen<elt_t>(1)
                                             << 7 << 3 << 9 << 5 << 11);
    this->emplace_back(Dimensions{2, 6}, gen<elt_t>(1)
                                             << 7 << 3 << 9 << 5 << 11 << 2
                                             << 14 << 6 << 18 << 10 << 22);

    this->emplace_back(Dimensions{2, 1}, gen<elt_t>(1) << 2);
    this->emplace_back(Dimensions{2, 3}, gen<elt_t>(1)
                                             << 7 << 3 << 9 << 5 << 11);
    this->emplace_back(Dimensions{4, 3}, gen<elt_t>(1)
                                             << 7 << 2 << 14 << 3 << 9 << 6
                                             << 18 << 5 << 11 << 10 << 22);
#undef add
  }
};

}  // namespace tensor_test
