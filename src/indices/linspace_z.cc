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

#include <tensor/tensor.h>
#include "linspace.hpp"

namespace tensor {

/**Vector of 'n' equally spaced numbers in the interval [min, max].*/
CTensor linspace(cdouble min, cdouble max, index n) {
  CTensor a(Dimensions{1}, Vector<cdouble>({min}));
  CTensor b(Dimensions{1}, Vector<cdouble>({max}));
  return linspace(a, b, n);
}

/**Tensor of 'n' equally spaced vectors in the interval [min, max].*/
CTensor linspace(const CTensor &min, const CTensor &max, index n) {
  return do_linspace(min, max, n);
}

}  // namespace tensor
