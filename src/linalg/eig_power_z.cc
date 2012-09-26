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

#include "eig_power.cc"

namespace linalg {

  /**Right eigenvalue and eigenvector with the largest absolute
     value, computed using the power method.

     \ingroup Linalg
  */
  tensor::cdouble
  eig_power_right(const CTensor &O, CTensor *vector, size_t iter)
  {
    return eig_power_loop(O, vector, 1, iter);
  }

  /**Left eigenvalue and eigenvector with the largest absolute
     value, computed using the power method.

     \ingroup Linalg
  */
  tensor::cdouble
  eig_power_left(const CTensor &O, CTensor *vector, size_t iter)
  {
    return eig_power_loop(O, vector, 0, iter);
  }

} // namespace linalg
