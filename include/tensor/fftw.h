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

#ifndef TENSOR_FFT_H
#define TENSOR_FFT_H

#include <tensor/config.h>
#include <tensor/tensor.h>

#ifndef TENSOR_USE_FFTW3
#error "libtensor was built without the fftw3 interface"
#endif

namespace tensor {

  /** The direction of the Fourier transform.
   *
   * FORWARD refers to the fft, BACKWARD to the inverse Fourier transform.
   */
  enum {
    FFTW_FORWARD = -1,
    FFTW_BACKWARD = 1
  };

  /** Calculates and returns the (unnormalized!) DFT or IDFT of the input, depending on the direction */
  const CTensor fftw(const CTensor& in, int direction);
  /** Calculates and returns the unnormalized DFT/IDFT only along a single dimension. */
  const CTensor fftw(const CTensor& in, index dim, int direction);
  /** Calculates the (unnormalized) DFT/IDFT of the input for all dimension where convert is true. */
  const CTensor fftw(const CTensor& in, const Booleans& convert, int direction);


  /** Like fftw, but overwrites the input. */
  void fftw_inplace(CTensor& in, int direction);
  /** Like fftw, but overwrites the input. */
  void fftw_inplace(CTensor& in, index dim, int direction);
  /** Like fftw, but overwrites the input. */
  void fftw_inplace(CTensor& in, const Booleans& convert, int direction);

  /** Shifts the half-planes of the input tensor to reorder the frequencies.
   * 
   * A direction of FFTW_FORWARD applied to the output of fftw() shifts the
   * half-planes such that the frequencies go from minimum to maximum. Setting
   * direction to FFTW_BACKWARD reverses this operation.
   *
   * Note that by definition, fftshift is not a terribly efficient function, and
   * the implementation is not the fastest, either, so if possible you should
   * not use this inside time-critical sections of the code.
   */
  const CTensor fftshift(const CTensor& in, int direction);
  /** fftshift along only one diemnsion. */
  const CTensor fftshift(const CTensor& in, index dim, int direction);
  /** fftshift along the given dimensions. */
  const CTensor fftshift(const CTensor& in, const Booleans& convert, int direction);

} // namespace tensor

#endif // TENSOR_FFT_H
