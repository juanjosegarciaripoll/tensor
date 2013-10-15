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

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <tensor/fftw.h>

namespace tensor_test {

  // Creates the matrix to do the DFT in the slow way.
  CTensor build_dft_transform(int N, int sign) {
    RTensor vector = linspace(0, N, N);
    vector = reshape(vector, N, 1);

    return exp(sign * (2*M_PI/N) * cdouble(0,1) * fold(vector, 1, vector, 1));
  }

  // DFT along all indices
  CTensor full_dft(const CTensor& input, int sign) {
    CTensor output(input);
    for (int dim = 0; dim < input.rank(); dim++) {
      CTensor trafo = build_dft_transform(output.dimension(dim), sign);
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

        CTensor ref_fourier = full_dft(input, -1);

        // test all variants of the forward FFT
        CTensor fourier = fftw(input);
        CTensor fft_forward = fftw(input, FFTW_FORWARD);
        CTensor fourier_inplace = input;
        CTensor fft_forward_inplace = input;
        fftw_inplace(fourier_inplace);
        fftw_inplace(fft_forward_inplace, FFTW_FORWARD);

        EXPECT_TRUE(all_equal(input.dimensions(), fourier.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fft_forward.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fourier_inplace.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fft_forward_inplace.dimensions()));

        EXPECT_TRUE(approx_eq(ref_fourier, fourier, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fft_forward, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fourier_inplace, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fft_forward_inplace, 1e-10));
      }
    }
  }

    TEST(FFTWTest, BackwardFFTTest) {
    for (int rank = 1; rank <= 3; rank++) {
      for (DimensionIterator iter(rank,8); iter; ++iter) {
        CTensor input = CTensor::random(*iter);

        if (input.size() == 0) {
          continue;
        }

        CTensor ref_fourier = full_dft(input, +1);

        // test all variants of the inverse FFT
        CTensor fourier = ifftw(input);
        CTensor fft_backward = fftw(input, FFTW_BACKWARD);
        CTensor fourier_inplace = input;
        CTensor fft_backward_inplace = input;
        ifftw_inplace(fourier_inplace);
        fftw_inplace(fft_backward_inplace, FFTW_BACKWARD);

        EXPECT_TRUE(all_equal(input.dimensions(), fourier.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fft_backward.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fourier_inplace.dimensions()));
        EXPECT_TRUE(all_equal(input.dimensions(), fft_backward_inplace.dimensions()));

        EXPECT_TRUE(approx_eq(ref_fourier, fourier, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fft_backward, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fourier_inplace, 1e-10));
        EXPECT_TRUE(approx_eq(ref_fourier, fft_backward_inplace, 1e-10));
      }
    }
  }
}