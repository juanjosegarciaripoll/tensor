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
void test_cgs_eye(int n) {
  Tensor A = Tensor::eye(n, n);
  for (int cols = 1; cols < n; cols++) {
    Tensor y = Tensor::random(n, cols);

    // If the initial state is a solution, this should be ok
    Tensor x = cgs(A, y, &y);
    EXPECT_CEQ(x, y);

    // Otherwise check a randomize initial state
    Tensor x_start = y + Tensor::random(n, cols) * 0.01;
    x = cgs(A, y, &x_start);
    EXPECT_CEQ(x, y);
  }
}

template <class Tensor>
void test_cgs_permuted_diagonal(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor B = Tensor::eye(n) + 0.125 * random_permutation(n);
    Tensor A = mmult(adjoint(B), B);
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(A, x);

    // If the initial state is a solution, this should be ok
    Tensor x0 = cgs(A, y, &x, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);

    // Otherwise check a randomize initial state
    Tensor x_start = x + Tensor::random(n, cols) * 0.02;
    x0 = cgs(A, y, &x_start, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);
  }
}

template <class Tensor>
void test_cgs_unitary(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor B =
        Tensor::eye(n) + 0.125 * random_unitary<typename Tensor::elt_t>(n);
    Tensor A = mmult(adjoint(B), B);
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(A, x);

    // If the initial state is a solution, this should be ok
    Tensor x0 = cgs(A, y, &x, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);

    // Otherwise check a randomize initial state
    Tensor x_start = x + Tensor::random(n, cols) * 0.02;
    x0 = cgs(A, y, &x_start, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);
  }
}

template <class Tensor>
const Tensor f(const Tensor &f) {
  return 1.5 * f;
}

template <class Tensor>
void test_cgs_functor(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor x = Tensor::random(n, cols);
    Tensor y = f<Tensor>(x);

    // If the initial state is a solution, this should be ok
    Tensor x0 = cgs(f<Tensor>, y, &x, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);

    // Otherwise check a randomize initial state
    Tensor x_start = x + Tensor::random(n, cols) * 0.02;
    x0 = cgs(f<Tensor>, y, &x_start, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);
  }
}

template <class Tensor>
void test_cgs_functor_1arg(int n) {
  for (int cols = 1; cols < n; cols++) {
    Tensor B =
        Tensor::eye(n) + 0.125 * random_unitary<typename Tensor::elt_t>(n);
    Tensor A = mmult(adjoint(B), B);
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(A, x);

    // If the initial state is a solution, this should be ok
    Tensor x0 = cgs([&A](const Tensor &x) -> Tensor { return mmult(A, x); }, y,
                    &x, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);

    // Otherwise check a randomize initial state
    Tensor x_start = x + Tensor::random(n, cols) * 0.02;
    x0 = cgs([&A](const Tensor &x) -> Tensor { return mmult(A, x); }, y,
             &x_start, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);
  }
}

template <class Tensor>
const Tensor f2(const Tensor &t, const Tensor &A, const Tensor &B) {
  return mmult(A, mmult(B, t));
}

template <class Tensor>
void test_cgs_functor_2arg(int n) {
  Tensor B = Tensor::eye(n) + 0.125 * random_unitary<typename Tensor::elt_t>(n);
  Tensor Bd = adjoint(B);
  for (int cols = 1; cols < n; cols++) {
    Tensor x = Tensor::random(n, cols);
    Tensor y = mmult(Bd, mmult(B, x));

    // If the initial state is a solution, this should be ok
    Tensor x0 = cgs(
        [&Bd, &B](const Tensor &x) -> Tensor { return mmult(Bd, mmult(B, x)); },
        y, &x, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);

    // Otherwise check a randomize initial state
    Tensor x_start = x + Tensor::random(n, cols) * 0.02;
    x0 = cgs(
        [&Bd, &B](const Tensor &x) -> Tensor { return mmult(Bd, mmult(B, x)); },
        y, &x_start, 0, 2 * EPSILON);
    EXPECT_CEQ(x, x0);
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RCgs, Eye) { test_over_integers(0, 22, test_cgs_eye<RTensor>); }

TEST(RCgs, PermutedDiagonal) {
  test_over_integers(1, 22, test_cgs_permuted_diagonal<RTensor>);
}

TEST(RCgs, Unitary) { test_over_integers(1, 22, test_cgs_unitary<RTensor>); }

TEST(RCgs, Functor) { test_over_integers(1, 22, test_cgs_functor<RTensor>); }

TEST(RCgs, Functor1Arg) {
  test_over_integers(1, 22, test_cgs_functor_1arg<RTensor>);
}

TEST(RCgs, Functor2Arg) {
  test_over_integers(1, 22, test_cgs_functor_2arg<RTensor>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CCgs, Eye) { test_over_integers(0, 22, test_cgs_eye<CTensor>); }

TEST(CCgs, PermutedDiagonal) {
  test_over_integers(1, 22, test_cgs_permuted_diagonal<CTensor>);
}

TEST(CCgs, Unitary) { test_over_integers(1, 22, test_cgs_unitary<CTensor>); }

TEST(CCgs, Functor) { test_over_integers(1, 22, test_cgs_functor<CTensor>); }

TEST(CCgs, Functor1Arg) {
  test_over_integers(1, 22, test_cgs_functor_1arg<CTensor>);
}

TEST(CCgs, Functor2Arg) {
  test_over_integers(1, 22, test_cgs_functor_2arg<CTensor>);
}

}  // namespace tensor_test
