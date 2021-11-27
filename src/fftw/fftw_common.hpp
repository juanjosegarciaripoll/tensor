// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2013 Juan Jose Garcia Ripoll

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

#include <tensor/config.h>
#ifdef TENSOR_USE_MKL
#include <fftw/fftw3.h>
#else
#include <fftw3.h>
#endif
#include <tensor/indices.h>

namespace tensor {

void do_fftw(fftw_complex* pin, fftw_complex* pout, const Indices& dims,
             int direction);
void do_fftw(fftw_complex* pin, fftw_complex* pout, int dim,
             const Indices& dims, int direction);
void do_fftw(fftw_complex* pin, fftw_complex* pout, const Booleans& convert,
             const Indices& dims, int direction);

}  // namespace tensor
