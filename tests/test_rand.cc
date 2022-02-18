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

#include <tensor/rand.h>
#include "rand/mt.h"
#include <gtest/gtest.h>

namespace {

using namespace ::tensor;

// The random integer type must have enough bits for integers.
TEST(RandTest, IntSize) {
  int int_size = sizeof(unsigned int);
  int rand_size = sizeof(rand_uint);
  EXPECT_LE(int_size, rand_size);
}

// The random integer type must have enough bits for longs.
TEST(RandTest, LongSize) {
  int long_size = sizeof(unsigned long);
  int rand_size = sizeof(rand_uint);
  EXPECT_LE(long_size, rand_size);
}

// Ensure that reseeding does indeed change the random number generator.
TEST(RandTest, Reseed) {
  int i = rand<int>();
  rand_reseed();
  int j = rand<int>();
  EXPECT_NE(i, j);
}

// Check that the distribution is balanced
TEST(RandTest, DoubleBalanced) {
  int total = 10000;
  double average = 0;
  for (int i = 1; i < total; ++i) {
    average += rand<double>() - 0.5;
  }
  average = std::abs(average) / total;
  EXPECT_GE(1 / sqrt((double)total), average);
}

// Check that the distribution has the appropriate standard deviation
TEST(RandTest, DoubleSigma) {
  int total = 100000;
  double sigma = 0;
  for (int i = 1; i < total; ++i) {
    double r = rand<double>() - 0.5;
    sigma += r * r;
  }
  sigma = sigma / total;
  double expected = (0.5 * 0.5 * 0.5) * 2.0 / 3.0;
  EXPECT_NEAR(expected, sigma, 1 / sqrt((double)total));
}

// Check that the distribution is balanced
TEST(RandTest, ComplexBalanced) {
  int total = 10000;
  cdouble average = 0;
  for (int i = 1; i < total; ++i) {
    average += rand<cdouble>() - to_complex(0.5, 0.5);
  }
  double re = std::abs(real(average)) / total;
  double im = std::abs(imag(average)) / total;
  EXPECT_GE(1 / sqrt((double)total), re);
  EXPECT_GE(1 / sqrt((double)total), im);
}

// Real and imaginary parts of the complex random number are uncorrelated
TEST(RandTest, ComplexUncorrelated) {
  int total = 10000;
  double corr = 0;
  for (int i = 1; i < total; ++i) {
    cdouble z = rand<cdouble>();
    corr += (real(z) - 0.5) * (imag(z) - 0.5);
  }
  corr = std::abs(corr) / total;
  EXPECT_GE(1 / sqrt((double)total), corr);
  EXPECT_GE(1 / sqrt((double)total), corr);
}

// Test random numbers in empty ranges
TEST(RandTest, IntEmpty) {
  EXPECT_EQ(0, rand<int>(0));
  EXPECT_EQ(-11, rand<int>(-11, -11));
}

// Test random numbers in empty ranges
TEST(RandTest, LongEmpty) {
  EXPECT_EQ(0, rand<long>(0));
  EXPECT_EQ(-13, rand<long>(-13, -13));
}

// Test random numbers in empty ranges
TEST(RandTest, ULongEmpty) {
  EXPECT_EQ(0, rand<unsigned long>(0));
  EXPECT_EQ(13, rand<unsigned long>(13, 13));
}

}  // namespace
