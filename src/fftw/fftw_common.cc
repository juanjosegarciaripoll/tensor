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

#include "fftw_common.hpp"


namespace tensor {

// basic FFTW along all degrees of freedom
void
do_fftw(fftw_complex* pin, fftw_complex* pout, const Indices& dims, int direction) {
  fftw_plan plan;
  int d = dims.size();

  if (d == 1) {
    plan = fftw_plan_dft_1d(dims[0], pin, pout, direction, FFTW_ESTIMATE);
  } else {
    int dimensions[d];
    for (int i = 0; i < d; i++) {
      dimensions[d - i - 1] = dims[i];
    }
    plan = fftw_plan_dft(d, dimensions, pin, pout, direction, FFTW_ESTIMATE);
  }
    
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void
do_fftw(fftw_complex* pin, fftw_complex* pout, index dim, const Indices& dims, int direction) {
  // Collect the dimensions for the transform
  fftw_iodim fft_dim;
  fftw_iodim loop_dims[dims.size()-1];
  index stride = 1;

  for (index i = 0; i < dim; i++) {
    loop_dims[i].n = dims[i];
    loop_dims[i].is = stride;
    loop_dims[i].os = stride;
    stride *= loop_dims[i].n;
  }

  fft_dim.n = dims[dim];
  fft_dim.is = stride;
  fft_dim.os = stride;
  stride *= fft_dim.n;

  for (index i = dim+1; i < dims.size(); i++) {
    loop_dims[i-1].n = dims[i];
    loop_dims[i-1].is = stride;
    loop_dims[i-1].os = stride;
    stride *= loop_dims[i-1].n;
  }

  // Now make a Guru plan and execute it.
  fftw_plan plan = fftw_plan_guru_dft(1, &fft_dim, dims.size()-1, loop_dims,
          pin, pout, direction, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void
do_fftw(fftw_complex* pin, fftw_complex* pout, const Booleans& convert, const Indices& dims, int direction) {
  // Collect the dimensions for the transform
  fftw_iodim fft_dims[dims.size()];
  fftw_iodim loop_dims[dims.size()];
  index fft_size = 0;
  index loop_size = 0;
  index stride = 1;

  for (index i = 0; i < dims.size(); i++) {
    if (convert[i] == false) {
      loop_dims[loop_size].n = dims[i];
      loop_dims[loop_size].is = stride;
      loop_dims[loop_size].os = stride;
      loop_size++;
    } else {
      fft_dims[fft_size].n = dims[i];
      fft_dims[fft_size].is = stride;
      fft_dims[fft_size].os = stride;
      fft_size++;
    }

    stride *= dims[i];
  }

  // Now make a Guru plan and execute it.
  fftw_plan plan = fftw_plan_guru_dft(fft_size, fft_dims, loop_size, loop_dims,
          pin, pout, direction, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

} // namespace tensor