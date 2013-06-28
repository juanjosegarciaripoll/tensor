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

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <tensor/fftw.h>

namespace tensor_test {

  // Creates the matrix to do the DFT in the slow way.
  CTensor build_dft_transform(int N) {
    RTensor vector(N, 1);

    for (int i = 0; i < N; i++) {
      vector.at_seq(i) = i;
    }

    return exp(-(2*M_PI/N) * cdouble(0,1) * fold(vector, 1, vector, 1));
  }

  // DFT along all indices
  CTensor full_dft(const CTensor& input) {
    CTensor output(input);
    for (int dim = 0; dim < input.rank(); dim++) {
      CTensor trafo = build_dft_transform(output.dimension(dim));
      output = foldin(trafo, 1, output, dim);
    }

    return output;
  }

  TEST(FFTWTest, ForwardFFTTest) {
    for (int rank = 1; rank <= 3; rank++) {
        for (DimensionIterator iter(rank,8); iter; ++iter) {
          CTensor input = CTensor::random(*iter);
          if (input.size() == 0) {
            continue;
          }

          CTensor fourier = fftw(input);
          CTensor ref_fourier = full_dft(input);

          EXPECT_TRUE(all_equal(input.dimensions(), fourier.dimensions()));
          EXPECT_TRUE(approx_eq(ref_fourier, fourier, 1e-10));
        }
    }
  }
}