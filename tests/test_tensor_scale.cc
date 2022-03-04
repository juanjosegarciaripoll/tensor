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

#include <tensor/tensor.h>
#include "loops.h"
#include "test_kron.hpp"

namespace tensor_test {

template <typename elt_t>
void test_scale_small() {
  elt_t zero = number_zero<elt_t>();
  Tensor<elt_t> A(Dimensions{2, 3, 4}, linspace(0, 23, 24));
  Tensor<elt_t> A1(Dimensions{2, 3, 4},
                   {0,  2,  2,  6,  4,  10, 6,  14, 8,  18, 10, 22,
                    12, 26, 14, 30, 16, 34, 18, 38, 20, 42, 22, 46});

  EXPECT_CEQ(A1, scale(A, 0, linspace(1, 2, 2)));
  EXPECT_CEQ(A1, scale(A, -3, linspace(1, 2, 2)));

  Tensor<elt_t> A2(Dimensions{2, 3, 4},
                   {0,  1,  4,  6,  12, 15, 6,  7,  16, 18, 30, 33,
                    12, 13, 28, 30, 48, 51, 18, 19, 40, 42, 66, 69});
  EXPECT_CEQ(A2, scale(A, 1, linspace(1, 3, 3)));
  EXPECT_CEQ(A2, scale(A, -2, linspace(1, 3, 3)));

  Tensor<elt_t> A4(Dimensions{2, 3, 4},
                   {0,  1,  2,  3,  4,  5,  12, 14, 16, 18, 20, 22,
                    36, 39, 42, 45, 48, 51, 72, 76, 80, 84, 88, 92});
  EXPECT_CEQ(A4, scale(A, 2, linspace(1, 4, 4)));
  EXPECT_CEQ(A4, scale(A, -1, linspace(1, 4, 4)));

  Tensor<elt_t> A6 = reshape(A, 6, 4);
  Tensor<elt_t> A7 = reshape(A4, 6, 4);
  EXPECT_CEQ(A7, scale(A6, 1, linspace(1, 4, 4)));
  EXPECT_CEQ(A7, scale(A6, -1, linspace(1, 4, 4)));

  Tensor<elt_t> A8 = reshape(A, 2, 12);
  Tensor<elt_t> A9 = reshape(A1, 2, 12);
  EXPECT_CEQ(A9, scale(A8, 0, linspace(1, 2, 2)));
}

TEST(TensorScale, RTensorSmall) { test_scale_small<double>(); }

TEST(TensorScale, CTensorSmall) { test_scale_small<cdouble>(); }

}  // namespace tensor_test
