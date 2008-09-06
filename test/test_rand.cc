// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/rand.h>
#include "rand/mt.h"
#include <gtest/gtest.h>

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
  EXPECT_GE(1/sqrt((double)total), average);
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
  double expected = (0.5*0.5*0.5) * 2.0 / 3.0;
  EXPECT_NEAR(expected, sigma, 1/sqrt((double)total));
}

// Check that the distribution is balanced
TEST(RandTest, ComplexBalanced) {
  int total = 10000;
  cdouble average = 0;
  for (int i = 1; i < total; ++i) {
    average += rand<cdouble>() - to_complex(0.5,0.5);
  }
  double re = std::abs(real(average)) / total;
  double im = std::abs(imag(average)) / total;
  EXPECT_LE(1/(double)total, re);
  EXPECT_LE(1/(double)total, im);
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
  EXPECT_LE(1/(double)total, corr);
  EXPECT_LE(1/(double)total, corr);
}
