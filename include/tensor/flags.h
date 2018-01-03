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

#ifndef TENSOR_FLAGS_H
#define TENSOR_FLAGS_H

#include <vector>

namespace tensor {

  class Flags {
  public:
    Flags();

    double get(unsigned int code) const;
    Flags &set(unsigned int code, double value);

    unsigned int create_key(double value);

  private:
    std::vector<double> _values;
  };

  extern Flags FLAGS;

  /**Debug level block_svd() routine.*/
  extern const unsigned int TENSOR_DEBUG_BLOCK_SVD;

} // namespace tensor

#endif
