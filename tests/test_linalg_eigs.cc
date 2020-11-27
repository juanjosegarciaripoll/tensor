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
#include <tensor/linalg.h>

namespace tensor_test {

using namespace tensor;
using namespace linalg;

//////////////////////////////////////////////////////////////////////
// EIGENVALUE DECOMPOSITIONS
//

template <class Matrix>
void test_eigs_eye(int n) {
  if (n == 0) {
#ifndef NDEBUG
    ASSERT_DEATH(eigs(Matrix::eye(n, n), 1, LargestMagnitude), ".*");
#endif
    return;
  }
  typedef typename Matrix::elt_t elt_t;
  Matrix A = Matrix::eye(n, n);
  Tensor<elt_t> V1 = Tensor<elt_t>::ones(n);
  for (int neig = 1; neig < std::min(n, 4); neig++) {
    Tensor<elt_t> U;
    for (int type = 0; type < LargestImaginary; type++) {
      CTensor E = eigs(A, LargestMagnitude, neig, &U);
      EXPECT_CEQ(sqrt((double)neig), norm2(U));
      EXPECT_EQ(2, U.rank());
      EXPECT_EQ(n, U.dimension(0));
      EXPECT_EQ(neig, U.dimension(1));
      EXPECT_EQ(neig, E.size());
      EXPECT_CEQ(CTensor::ones(igen << neig), E);
    }
  }
}

template <class Matrix>
void test_eigs_permuted_diagonal(int n) {
  typedef typename Matrix::elt_t elt_t;
  RTensor p = random_permutation(n, n);
  RTensor pinv = adjoint(p);
  RTensor d = diag(linspace((double)1.0, n, n), 0);

  Tensor<elt_t> e1 = RTensor::zeros(igen << n);
  e1.at(0) = 1.0;
  e1 = mmult(pinv, e1);

  Tensor<elt_t> en = RTensor::zeros(igen << n);
  en.at(n - 1) = 1.0;
  en = mmult(pinv, en);

  typedef typename Matrix::elt_t elt_t;
  Matrix A = mmult(pinv, mmult(d, p));
  Tensor<elt_t> U;

  CTensor E = eigs(A, SmallestMagnitude, 1, &U);
  EXPECT_EQ(2, U.rank());
  EXPECT_EQ(n, U.dimension(0));
  EXPECT_EQ(1, U.dimension(1));
  EXPECT_EQ(1, E.size());
  EXPECT_CEQ(number_one<cdouble>(), E(0));
  EXPECT_CEQ(1.0, abs(fold(e1, 0, U, 0))(0));

  E = eigs(A, LargestMagnitude, 1, &U);
  EXPECT_EQ(2, U.rank());
  EXPECT_EQ(n, U.dimension(0));
  EXPECT_EQ(1, U.dimension(1));
  EXPECT_EQ(1, E.size());
  EXPECT_TRUE(simeq(to_complex((double)n), E(0), 20 * EPSILON));
  EXPECT_CEQ(1.0, abs(fold(en, 0, U, 0))(0));
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RArpackTest, EigsEye) {
  test_over_integers(0, 22, test_eigs_eye<RTensor>);
}

TEST(RArpackTest, EigsPermutedDiagonal) {
  test_over_integers(1, 22, test_eigs_permuted_diagonal<RTensor>);
}

TEST(RArpackTest, EigsRSparseEye) {
  test_over_integers(0, 22, test_eigs_eye<RSparse>);
}

TEST(RArpackTest, EigsRSparsePermutedDiagonal) {
  test_over_integers(1, 22, test_eigs_permuted_diagonal<RTensor>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CArpackTest, EigsEye) {
  test_over_integers(0, 22, test_eigs_eye<CTensor>);
}

TEST(CArpackTest, EigsPermutedDiagonal) {
  test_over_integers(1, 22, test_eigs_permuted_diagonal<CTensor>);
}

TEST(CArpackTest, EigsCSparseEye) {
  test_over_integers(0, 22, test_eigs_eye<CSparse>);
}

TEST(CArpackTest, EigsCSparsePermutedDiagonal) {
  test_over_integers(1, 22, test_eigs_permuted_diagonal<CTensor>);
}

}  // namespace tensor_test
