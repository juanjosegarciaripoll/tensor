// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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

/* MULTIDIMENSIONAL FAST FOURIER TRANSFORM */

#include <tensor/fftw.h>
#include "fftw_common.hpp"

namespace tensor {

CTensor fftw(const CTensor &in, int direction) {
  CTensor out(in.dimensions());

  fftw_complex *pin = const_cast<fftw_complex *>(
      reinterpret_cast<const fftw_complex *>(in.begin()));
  fftw_complex *pout = reinterpret_cast<fftw_complex *>(out.begin());
  do_fftw(pin, pout, in.dimensions(), direction);

  return out;
}

CTensor fftw(const CTensor &in, index dim, int direction) {
  assert(dim >= 0 && dim < in.rank());
  CTensor out(in.dimensions());

  fftw_complex *pin = const_cast<fftw_complex *>(
      reinterpret_cast<const fftw_complex *>(in.begin()));
  fftw_complex *pout = reinterpret_cast<fftw_complex *>(out.begin());
  do_fftw(pin, pout, static_cast<int>(dim), in.dimensions(), direction);

  return out;
}

CTensor fftw(const CTensor &in, const Booleans &convert, int direction) {
  assert(convert.ssize() == in.rank());
  CTensor out(in.dimensions());

  fftw_complex *pin = const_cast<fftw_complex *>(
      reinterpret_cast<const fftw_complex *>(in.begin()));
  fftw_complex *pout = reinterpret_cast<fftw_complex *>(out.begin());
  do_fftw(pin, pout, convert, in.dimensions(), direction);

  return out;
}

}  // namespace tensor
