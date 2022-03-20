/*
    Copyright (c) 2010-2013 Juan Jose Garcia Ripoll

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

#include <tensor/fftw.h>
#include "fftw_common.hpp"

namespace tensor {

void fftw_inplace(CTensor& in, int direction) {
  auto pin = reinterpret_cast<fftw_complex*>(in.begin());
  do_fftw(pin, pin, in.dimensions(), direction);
}

void fftw_inplace(CTensor& in, index dim, int direction) {
  tensor_assert(dim >= 0 && dim < in.rank());
  auto pin = reinterpret_cast<fftw_complex*>(in.begin());
  do_fftw(pin, pin, static_cast<int>(dim), in.dimensions(), direction);
}

void fftw_inplace(CTensor& in, const Booleans& convert, int direction) {
  tensor_assert(convert.ssize() == in.rank());
  auto pin = reinterpret_cast<fftw_complex*>(in.begin());
  do_fftw(pin, pin, convert, in.dimensions(), direction);
}

}  // namespace tensor
