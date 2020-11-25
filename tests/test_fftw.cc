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
    RTensor vector = linspace(0, N-1, N);
    vector = reshape(vector, N, 1);

    return exp(sign * (2*M_PI/N) * cdouble(0,1) * fold(vector, 1, vector, 1));
  }

  // DFT along only one index.
  CTensor partial_dft(const CTensor& input, int idx, int sign) {
    CTensor trafo = build_dft_transform(input.dimension(idx), sign);
    return foldin(trafo, 1, input, idx);
  }

  // DFT along multiple indices.
  CTensor partial_dft(const CTensor& input, const Booleans& convert, int sign) {
    CTensor output(input);
    for (int dim = 0; dim < input.rank(); dim++) {
      if (convert[dim] == true) {
        output = partial_dft(output, dim, sign);
      }
    }

    return output;
  }

  // DFT along all indices
  CTensor full_dft(const CTensor& input, int sign) {
    CTensor output(input);
    for (int dim = 0; dim < input.rank(); dim++) {
      output = partial_dft(output, dim, sign);
    }

    return output;
  }

  // Sets up the permutation matrix to use for fft shifts.
  // Slow, but rather transparent and simple.
  RTensor build_permutation_matrix(int N, int direction) {
    RTensor permute = RTensor::zeros(N, N);

    // even N is trivial, just swap the two half-spaces; both permutations are identical.
    if (N % 2 == 0) {
      for (int i = 0; i < N/2; i++) {
        permute.at(i,N/2+i) = 1;
        permute.at(N/2+i, i) = 1;
      }

      return permute;
    }

    // odd N is a bit more complicated
    int center = (N-1)/2;
    permute.at(center, 0) = 1;
    for (int i = 0; i < center; i++) {
      permute.at(i, center+1+i) = 1;
      permute.at(center+1+i, i+1) = 1;
    }

    if (direction == FFTW_BACKWARD) {
      permute = transpose(permute);
    }

    return permute;
  }

  // fftshift along a single dimension
  CTensor single_fft_shift(const CTensor& input, int dim, int direction) {
    CTensor permutation = build_permutation_matrix(input.dimension(dim), direction);

    return foldin(permutation, 1, input, dim);
  }

  // fftshift along a couple of dimensions
  CTensor multiple_fft_shift(const CTensor& input, Booleans convert, int direction) {
    CTensor output(input);
    for (int i = 0; i < input.rank(); i++) {
      if (convert[i] == true) {
        output = single_fft_shift(output, i, direction);
      }
    }

    return output;
  }

  // fftshift along all dimensions
  CTensor all_fft_shift(const CTensor& input, int direction) {
    CTensor output(input);
    for (int i = 0; i < input.rank(); i++) {
      output = single_fft_shift(output, i, direction);
    }

    return output;
  }


  TEST(FFTWTest, OutOfPlaceFFTTest) {
    for (int rank = 1; rank <= 3; rank++) {
      for (DimensionIterator iter(rank,6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);
        if (input.size() == 0) {
          continue;
        }

        // first, test the fftw along all dimensions
        CTensor ref_fft = full_dft(input, -1);
        CTensor ref_ifft = full_dft(input, +1);

        EXPECT_TRUE(approx_eq(ref_fft, fftw(input, FFTW_FORWARD), 1e-10));
        EXPECT_TRUE(approx_eq(ref_ifft, fftw(input, FFTW_BACKWARD), 1e-10));

        for (tensor::index dim = 0; dim < rank; dim++) {
          // test the fftw along only a single dimension
          ref_fft = partial_dft(input, dim, -1);
          ref_ifft = partial_dft(input, dim, +1);

          EXPECT_TRUE(approx_eq(ref_fft, fftw(input, dim, FFTW_FORWARD), 1e-10));
          EXPECT_TRUE(approx_eq(ref_ifft, fftw(input, dim, FFTW_BACKWARD), 1e-10));
        }
        for (BooleansIterator biter(rank); biter; ++biter) {
          ref_fft = partial_dft(input, *biter, -1);
          ref_ifft = partial_dft(input, *biter, +1);

          EXPECT_TRUE(approx_eq(ref_fft, fftw(input, *biter, FFTW_FORWARD), 1e-10));
          EXPECT_TRUE(approx_eq(ref_ifft, fftw(input, *biter, FFTW_BACKWARD), 1e-10));
        }
      }
    }
  }

  TEST(FFTWTest, OutOfPlaceDeathTest) {
    for (int rank = 1; rank < 3; rank++) {
      for (DimensionIterator iter(rank,6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);

        ASSERT_DEATH(fftw(input, -1, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw(input, rank, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw(input, Booleans(rank+1), FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw(input, Booleans(rank-1), FFTW_FORWARD), ".*");
      }
    }
  }

  TEST(FFTWTest, InPlaceFFTTest) {
    for (int rank = 1; rank <= 3; rank++) {
      for (DimensionIterator iter(rank,6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);
        if (input.size() == 0) {
          continue;
        }

        // first, test the fftw along all dimensions
        CTensor ref_fft = full_dft(input, -1);
        CTensor ref_ifft = full_dft(input, +1);

        CTensor inplace = input;
        fftw_inplace(inplace, FFTW_FORWARD);
        EXPECT_TRUE(approx_eq(ref_fft, inplace, 1e-10));

        inplace = input;
        fftw_inplace(inplace, FFTW_BACKWARD);
        EXPECT_TRUE(approx_eq(ref_ifft, inplace, 1e-10));

        for (tensor::index dim = 0; dim < rank; dim++) {
          // test the fftw along only a single dimension
          ref_fft = partial_dft(input, dim, -1);
          ref_ifft = partial_dft(input, dim, +1);

          inplace = input;
          fftw_inplace(inplace, dim, FFTW_FORWARD);
          EXPECT_TRUE(approx_eq(ref_fft, inplace, 1e-10));

          inplace = input;
          fftw_inplace(inplace, dim, FFTW_BACKWARD);
          EXPECT_TRUE(approx_eq(ref_ifft, inplace, 1e-10));
        }
        for (BooleansIterator biter(rank); biter; ++biter) {
          ref_fft = partial_dft(input, *biter, -1);
          ref_ifft = partial_dft(input, *biter, +1);

          inplace = input;
          fftw_inplace(inplace, *biter, FFTW_FORWARD);
          EXPECT_TRUE(approx_eq(ref_fft, inplace, 1e-10));

          inplace = input;
          fftw_inplace(inplace, *biter, FFTW_BACKWARD);
          EXPECT_TRUE(approx_eq(ref_ifft, inplace, 1e-10));
        }
      }
    }
  }

  TEST(FFTWTest, InPlaceDeathTest) {
    for (int rank = 1; rank < 3; rank++) {
      for (DimensionIterator iter(rank,6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);

        ASSERT_DEATH(fftw_inplace(input, -1, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw_inplace(input, rank, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw_inplace(input, Booleans(rank+1), FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftw_inplace(input, Booleans(rank-1), FFTW_FORWARD), ".*");
      }
    }
  }

  TEST(FFTWTest, fftShiftTest) {
    for (int rank = 1; rank < 4; rank++) {
      for (DimensionIterator iter(rank, 6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);
        if (input.size() == 0) {
          continue;
        }

        EXPECT_TRUE(approx_eq(all_fft_shift(input, FFTW_FORWARD),
                fftshift(input, FFTW_FORWARD)));
        EXPECT_TRUE(approx_eq(all_fft_shift(input, FFTW_BACKWARD),
                fftshift(input, FFTW_BACKWARD)));

        for (tensor::index dim = 0; dim < rank; dim++) {
          EXPECT_TRUE(approx_eq(single_fft_shift(input, dim, FFTW_FORWARD),
                  fftshift(input, dim, FFTW_FORWARD)));
          EXPECT_TRUE(approx_eq(single_fft_shift(input, dim, FFTW_BACKWARD),
                  fftshift(input, dim, FFTW_BACKWARD)));
        }
        for (BooleansIterator biter(rank); biter; ++biter) {
          EXPECT_TRUE(approx_eq(multiple_fft_shift(input, *biter, FFTW_FORWARD),
                  fftshift(input, *biter, FFTW_FORWARD)));
          EXPECT_TRUE(approx_eq(multiple_fft_shift(input, *biter, FFTW_BACKWARD),
                  fftshift(input, *biter, FFTW_BACKWARD)));
        }
      }
    }
  }

  TEST(FFTWTest, fftShiftDeathTest) {
    for (int rank = 1; rank < 3; rank++) {
      for (DimensionIterator iter(rank,6); iter; ++iter) {
        CTensor input = CTensor::random(*iter);

        ASSERT_DEATH(fftshift(input, -1, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftshift(input, rank, FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftshift(input, Booleans(rank+1), FFTW_FORWARD), ".*");
        ASSERT_DEATH(fftshift(input, Booleans(rank-1), FFTW_FORWARD), ".*");
      }
    }
  }
}
