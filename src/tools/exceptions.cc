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

#include <string>
#include <sstream>
#include <tensor/exceptions.h>
#include <tensor/tensor_blas.h>
#include <tensor/tensor.h>
#include <tensor/io.h>

namespace blas {

blas_integer_overflow::blas_integer_overflow()
    : std::out_of_range(
          "Tensor size exceeds limits supported by BLAS implementation.") {}

}  // namespace blas

namespace tensor {

static std::string dimensions_mismatch_message(const Dimensions &d1,
                                               const Dimensions &d2) {
  std::ostringstream out;
  out << "Unable to perform binary operation among tensors with dimensions\n"
      << d1 << " and " << d2;
  return out.str();
}

dimensions_mismatch::dimensions_mismatch(const Dimensions &d1,
                                         const Dimensions &d2)
    : std::out_of_range(dimensions_mismatch_message(d1, d2)) {}

static std::string dimensions_mismatch_message(const Dimensions &d1,
                                               const Dimensions &d2,
                                               index which1, index which2) {
  std::ostringstream out;
  out << "Unable to perform binary operation among tensors with dimensions\n"
      << d1 << " and " << d2 << "\nbecause indices " << which1 << " and "
      << which2 << " differ or are zero";
  return out.str();
}

dimensions_mismatch::dimensions_mismatch(const Dimensions &d1,
                                         const Dimensions &d2, index which1,
                                         index which2)
    : std::out_of_range(dimensions_mismatch_message(d1, d2, which1, which2)) {}

}  // namespace tensor
