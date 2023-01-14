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

    this->push_back(Tensor2D<elt_t>({{1}}));
    this->push_back(Tensor2D<elt_t>({{-2}}));
    this->push_back(Tensor2D<elt_t>({{-2}}));

    this->push_back(Tensor2D<elt_t>({{1,2}}));
    this->push_back(Tensor2D<elt_t>({{1, 3, 5}, {7, 9, 11}}));
    this->push_back(Tensor2D<elt_t>({{1, 3, 5, 2, 6, 10}, {7, 9, 11, 14, 18, 22}}));

    this->push_back(Tensor2D<elt_t>({{1}, {2}}));
    this->push_back(Tensor2D<elt_t>({{1, 3, 5}, {7, 9, 11}}));
    this->push_back(Tensor2D<elt_t>({{1, 3, 5}, {7, 9, 11}, {2, 6, 10}, {14, 18, 22}}));
#undef add
  }
};

}  // namespace tensor_test
