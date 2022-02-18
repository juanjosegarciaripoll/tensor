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

template <class Tensor>
void test_solve_eye(int n) {
  Tensor A = Tensor::eye(n, n);
  for (int cols = 1; cols < n; cols++) {
    Tensor y = Tensor::random(n, cols);
    Tensor x = solve(A, y);
    EXPECT_CEQ(x, y);
  }
}

template <class Tensor>
void test_solve_permuted_diagonal(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor A = random_permutation(n);
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(A, x);
    Tensor x0 = solve(A, y);
    EXPECT_CEQ(x, x0);
  }
}

template <class Tensor>
void test_solve_unitary(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor A = random_unitary<typename Tensor::elt_t>(n);
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(A, x);
    Tensor x0 = solve(A, y);
    EXPECT_CEQ(x, x0);
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RSolve, Eye) { test_over_integers(0, 22, test_solve_eye<RTensor>); }

TEST(RSolve, PermutedDiagonal) {
  test_over_integers(1, 22, test_solve_permuted_diagonal<RTensor>);
}

TEST(RSolve, Unitary) {
  test_over_integers(1, 22, test_solve_unitary<RTensor>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CSolve, Eye) { test_over_integers(0, 22, test_solve_eye<CTensor>); }

TEST(CSolve, PermutedDiagonal) {
  test_over_integers(1, 22, test_solve_permuted_diagonal<CTensor>);
}

TEST(CSolve, Unitary) {
  test_over_integers(1, 22, test_solve_unitary<CTensor>);
}

}  // namespace tensor_test
