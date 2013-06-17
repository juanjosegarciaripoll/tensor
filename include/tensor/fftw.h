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

#ifndef TENSOR_FFT_H
#define TENSOR_FFT_H

#include <tensor/config.h>
#include <tensor/tensor.h>

#ifndef TENSOR_USE_FFTW3
#error "libtensor was built without the fftw3 interface"
#endif

namespace tensor {

  const CTensor ifftw(const CTensor &in);
  const CTensor fftw(const CTensor &in, int direction = +1);

  void fftw_inplace(CTensor *in, int direction = +1);
  void ifftw_inplace(CTensor *in);

} // namespace tensor

#endif // TENSOR_FFT_H
