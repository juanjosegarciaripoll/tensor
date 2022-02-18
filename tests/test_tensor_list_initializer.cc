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
#include <tensor/tensor.h>

namespace tensor_test {

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RTensorTest, RTensor1DEmptyListConstructor) {
  RTensor A = {};
  EXPECT_EQ(A.size(), 0);
  EXPECT_EQ(A.rank(), 0);
}

TEST(RTensorTest, RTensor1DListConstructor) {
  RTensor A = {1.0, 2.0, 3.0};
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A.rank(), 1);
  EXPECT_EQ(A.dimension(0), 3);
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 2.0);
  EXPECT_EQ(A[2], 3.0);
}

TEST(RTensorTest, RTensor2DListConstructor) {
  RTensor A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_EQ(A.size(), 6);
  EXPECT_EQ(A.rank(), 2);
  EXPECT_EQ(A.dimension(0), 2);
  EXPECT_EQ(A.dimension(1), 3);
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 4.0);
  EXPECT_EQ(A[2], 2.0);
  EXPECT_EQ(A[3], 5.0);
  EXPECT_EQ(A[4], 3.0);
  EXPECT_EQ(A[5], 6.0);
}

RTensor rtensor_failed_2d_list_initialization() {
  // Length of sublists do not match
  RTensor A = {{1.0, 2.0}, {3.0, 4.0, 5.0}};
  return A;
}

TEST(RTensorTest, RTensor2DListFailedConstructor) {
  EXPECT_THROW(rtensor_failed_2d_list_initialization(), std::out_of_range);
}

TEST(RTensorTest, RTensor3DListConstructor) {
  RTensor A = {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};
  EXPECT_EQ(A.size(), 12);
  EXPECT_EQ(A.rank(), 3);
  EXPECT_EQ(A.dimension(0), 2);
  EXPECT_EQ(A.dimension(1), 3);
  EXPECT_EQ(A.dimension(2), 2);
  EXPECT_EQ(A(0, 0, 0), 1.0);
  EXPECT_EQ(A(0, 0, 1), 2.0);
  EXPECT_EQ(A(0, 1, 0), 3.0);
  EXPECT_EQ(A(0, 1, 1), 4.0);
  EXPECT_EQ(A(0, 2, 0), 5.0);
  EXPECT_EQ(A(0, 2, 1), 6.0);
  EXPECT_EQ(A(1, 0, 0), 7.0);
  EXPECT_EQ(A(1, 0, 1), 8.0);
  EXPECT_EQ(A(1, 1, 0), 9.0);
  EXPECT_EQ(A(1, 1, 1), 10.0);
  EXPECT_EQ(A(1, 2, 0), 11.0);
  EXPECT_EQ(A(1, 2, 1), 12.0);
}

RTensor rtensor_failed_3d_list_initialization() {
  // Length of sublists do not match
  RTensor A = {{{1.0, 2.0}, {2.0, 3.0}}, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};
  return A;
}

TEST(RTensorTest, RTensor3DListFailedConstructor) {
  EXPECT_THROW(rtensor_failed_3d_list_initialization(), std::out_of_range);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CTensorTest, CTensor1DEmptyListConstructor) {
  CTensor A = {};
  EXPECT_EQ(A.size(), 0);
  EXPECT_EQ(A.rank(), 0);
}

TEST(CTensorTest, CTensor1DListConstructor) {
  CTensor A = {cdouble(1.0), 2.0, 3.0};
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A.rank(), 1);
  EXPECT_EQ(A.dimension(0), 3);
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 2.0);
  EXPECT_EQ(A[2], 3.0);
}

TEST(CTensorTest, CTensor2DListConstructor) {
  CTensor A = {{cdouble(1.0), 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_EQ(A.size(), 6);
  EXPECT_EQ(A.rank(), 2);
  EXPECT_EQ(A.dimension(0), 2);
  EXPECT_EQ(A.dimension(1), 3);
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 4.0);
  EXPECT_EQ(A[2], 2.0);
  EXPECT_EQ(A[3], 5.0);
  EXPECT_EQ(A[4], 3.0);
  EXPECT_EQ(A[5], 6.0);
}

CTensor ctensor_failed_2d_list_initialization() {
  CTensor A = {{cdouble(1.0), 2.0}, {3.0, 4.0, 5.0}};
  return A;
}

TEST(CTensorTest, CTensor2DListFailedConstructor) {
  EXPECT_THROW(ctensor_failed_2d_list_initialization(), std::out_of_range);
}

TEST(CTensorTest, CTensor3DListConstructor) {
  CTensor A = {{{cdouble(1), 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};
  EXPECT_EQ(A.size(), 12);
  EXPECT_EQ(A.rank(), 3);
  EXPECT_EQ(A.dimension(0), 2);
  EXPECT_EQ(A.dimension(1), 3);
  EXPECT_EQ(A.dimension(2), 2);
  EXPECT_EQ(A(0, 0, 0), 1.0);
  EXPECT_EQ(A(0, 0, 1), 2.0);
  EXPECT_EQ(A(0, 1, 0), 3.0);
  EXPECT_EQ(A(0, 1, 1), 4.0);
  EXPECT_EQ(A(0, 2, 0), 5.0);
  EXPECT_EQ(A(0, 2, 1), 6.0);
  EXPECT_EQ(A(1, 0, 0), 7.0);
  EXPECT_EQ(A(1, 0, 1), 8.0);
  EXPECT_EQ(A(1, 1, 0), 9.0);
  EXPECT_EQ(A(1, 1, 1), 10.0);
  EXPECT_EQ(A(1, 2, 0), 11.0);
  EXPECT_EQ(A(1, 2, 1), 12.0);
}

CTensor ctensor_failed_3d_list_initialization() {
  CTensor A = {{{cdouble(1.0), 2.0}, {2.0, 3.0}},
               {{1.0, 2.0, 3.0}, {4.0, 5.0}}};
  return A;
}

TEST(CTensorTest, CTensor3DListFailedConstructor) {
  EXPECT_THROW(ctensor_failed_3d_list_initialization(), std::out_of_range);
}

}  // namespace tensor_test
