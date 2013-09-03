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

#include <tensor/fftw.h>
#include <fftw3.h>

namespace tensor {

  void fftw_inplace(CTensor& in)
  {
    fftw_inplace(in, FFTW_FORWARD);
  }

  void
  ifftw_inplace(CTensor& in)
  {
    fftw_inplace(in, FFTW_BACKWARD);
  }

  void
  fftw_inplace(CTensor& in, int sense)
  {
    int d = in.rank();

    fftw_complex *pin =
            const_cast<fftw_complex*>
            (reinterpret_cast<const fftw_complex*> (in.begin()));
    fftw_plan plan;

    if (d == 1) {
      plan = fftw_plan_dft_1d(in.size(), pin, pin, sense, FFTW_ESTIMATE);
    } else {
      int dimensions[d];
      for (int i = 0; i < d; i++) {
        dimensions[d - i - 1] = in.dimension(i);
      }
      plan = fftw_plan_dft(d, dimensions, pin, pin, sense, FFTW_ESTIMATE);
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

} // namespace tensor
