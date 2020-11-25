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

#include "slow_fold.cc"

namespace tensor_test {

  //////////////////////////////////////////////////////////////////////
  // MATRIX MULTIPLICATION
  //

  template<typename n1, typename n2>
  void test_fold(index max_dim) {
    index count = 0;
    for (int rankA = 1; rankA <= 3; rankA++) {
      for (int rankB = 1; rankB <= 3; rankB++) {
        for (DimensionIterator dA(rankA,max_dim); dA; ++dA) {
          Tensor<n1> A(*dA);
          for (DimensionIterator dB(rankB,max_dim); dB; ++dB) {
            Tensor<n2> B(*dB);
            for (int i = 0; i < A.rank(); i++) {
              if (!A.dimension(i)) continue;
              for (int j = 0; j < B.rank(); j++) {
                if (A.dimension(i) == B.dimension(j)) {
                  A.randomize();
                  B.randomize();
                  index expected_size = A.size()*B.size()/A.dimension(i)/B.dimension(j);
                  Tensor<typename Binop<n1,n2>::type>
                    AB = fold(A,i,B,j),
                    sAB = slow_fold(A,i,B,j);
                  // Compare optimized routine with safer, slow ones
                  EXPECT_TRUE(all_equal(AB.dimensions(), sAB.dimensions()));
                  EXPECT_EQ(expected_size, AB.size());
                  EXPECT_TRUE(approx_eq(AB, sAB));
                  // Negative indices conditions
                  EXPECT_TRUE(all_equal(AB, fold(A, i - rankA, B, j)));
                  EXPECT_TRUE(all_equal(AB, fold(A, i, B, j - rankB)));
                  EXPECT_TRUE(all_equal(AB, fold(A, i - rankA, B, j - rankB)));
                  // Original tensors are not changed
                  unique(A);
                  unique(B);
                  if (count % 1000 == 0) {
                    std::cout << '.' << std::flush;
                  }
                  if ((++count) % 60000 == 0) {
                    std::cout << std::endl;
                  }
                }
              }
            }
          }
        }
      }
    }
    std::cout << std::endl;
  }

  template<typename n1, typename n2>
  void test_fold_death() {
    for (int rankA = 1; rankA <= 4; rankA++) {
      for (int rankB = 1; rankB <= 4; rankB++) {
        for (int i = 0; i < rankA; i++) {
          for (int j = 0; j < rankB; j++) {
            Indices dA(rankA), dB(rankB);
            std::fill(dA.begin(), dA.end(), 1);
            std::fill(dB.begin(), dB.end(), 1);
            dA.at(i) = 0;
            dB.at(j) = 0;
            Tensor<n1> A(dA);
            Tensor<n2> B(dB);
            ASSERT_DEATH(fold(A, i, B, j), ".*");
          }
        }
      }
    }
  }
 

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

#define MATRIX_MAX_DIM 8

  TEST(FoldTest, FoldDoubleDoubleTest) {
    test_fold<double,double>(MATRIX_MAX_DIM);
  }

  TEST(FoldTest, FoldDoubleDoubleDeathTest) {
    test_fold_death<double,double>();
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(FoldTest, FoldCdoubleCdoubleTest) {
    test_fold<cdouble,cdouble>(MATRIX_MAX_DIM);
  }

  TEST(FoldTest, FoldCdoubleCdoubleDeathTest) {
    test_fold_death<cdouble,cdouble>();
  }

} // namespace tensor_test
