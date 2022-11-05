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

template <typename elt_t>
void test_empty_constructor() {
  {
    SCOPED_TRACE("0D");
    Tensor<elt_t> P;
    EXPECT_EQ(0, P.size());
    EXPECT_EQ(0, P.rank());
    // In a default constructor, the content of the pointer is unknown
    // hence we do not know the reference count.
    /* EXPECT_EQ(1, P.ref_count()); */
    EXPECT_EQ(P.cend(), P.cbegin());
  }
  {
    SCOPED_TRACE("1D");
    auto P = Tensor<elt_t>::empty(0);
    EXPECT_EQ(0, P.size());
    EXPECT_EQ(1, P.rank());
    EXPECT_EQ(1, P.ref_count());
    EXPECT_EQ(P.cend(), P.cbegin());
  }
  {
    SCOPED_TRACE("2D");
    auto P = Tensor<elt_t>::empty(1, 0);
    EXPECT_EQ(0, P.size());
    EXPECT_EQ(2, P.rank());
    EXPECT_EQ(1, P.ref_count());
    EXPECT_EQ(P.cend(), P.cbegin());
  }
}

template <typename elt_t>
void test_copy_constructor(Tensor<elt_t> &P) {
  Tensor<elt_t> P2(P);
  unchanged(P2, P);
}

// Test proper work of dimension querying routines>
//	- they must return individual dimensions
//	- must not change actual data
//
template <typename elt_t>
void test_dims(Tensor<elt_t> &P) {
  Dimensions d = P.dimensions();

  Indices::elt_t a1, a2, a3, a4, a5, a6;
  ASSERT_LE(P.rank(), 6);
  EXPECT_EQ(d.rank(), P.rank());
  switch (P.rank()) {
    case 6: {
      P.get_dimensions(&a1, &a2, &a3, &a4, &a5, &a6);
      EXPECT_EQ(a1 * a2 * a3 * a4 * a5 * a6, P.size());
      break;
    }
    case 5: {
      P.get_dimensions(&a1, &a2, &a3, &a4, &a5);
      EXPECT_EQ(a1 * a2 * a3 * a4 * a5, P.size());
      break;
    }
    case 4: {
      P.get_dimensions(&a1, &a2, &a3, &a4);
      EXPECT_EQ(a1 * a2 * a3 * a4, P.size());
      break;
    }
    case 3: {
      P.get_dimensions(&a1, &a2, &a3);
      EXPECT_EQ(a1 * a2 * a3, P.size());
      break;
    }
    case 2: {
      P.get_dimensions(&a1, &a2);
      EXPECT_EQ(a1 * a2, P.size());
      break;
    }
    case 1: {
      P.get_dimensions(&a1);
      EXPECT_EQ(a1, P.size());
      break;
    }
    case 0: {
      EXPECT_EQ(0, P.size());
      break;
    }
  }
  switch (P.rank()) {
    case 6:
      EXPECT_EQ(a6, d[5]);
      EXPECT_EQ(a6, P.dimension(5));
    case 5:
      EXPECT_EQ(a5, d[4]);
      EXPECT_EQ(a5, P.dimension(4));
    case 4:
      EXPECT_EQ(a4, d[3]);
      EXPECT_EQ(a4, P.dimension(3));
    case 3:
      EXPECT_EQ(a3, d[2]);
      EXPECT_EQ(a3, P.dimension(2));
    case 2:
      EXPECT_EQ(a2, d[1]);
      EXPECT_EQ(a2, P.dimension(1));
      EXPECT_EQ(a2, P.columns());
    case 1:
      EXPECT_EQ(a1, d[0]);
      EXPECT_EQ(a1, P.dimension(0));
      EXPECT_EQ(a1, P.rows());
  }
}

// Verify that get_dimensions signals an error for a wrong number of
// arguments
template <typename elt_t>
void test_get_dimensions_errors() {
#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
  for (int i = 0; i <= 6; i++) {
    Indices dimensions(i);
    std::fill(dimensions.begin(), dimensions.end(), (Indices::elt_t)2);

    auto P = Tensor<elt_t>::empty(Dimensions(dimensions));

    ASSERT_THROW_DEBUG(P.dimension(-1), ::tensor::invalid_assertion);
    ASSERT_THROW_DEBUG(P.dimension(i), ::tensor::invalid_assertion);

    for (size_t j = 0; j != i; j++) {
      Indices::elt_t a[6];
      switch (j) {
        case 6:
          ASSERT_THROW_DEBUG(
              P.get_dimensions(a, a + 1, a + 2, a + 3, a + 4, a + 5),
              ::tensor::invalid_assertion);
          break;
        case 5:
          ASSERT_THROW_DEBUG(P.get_dimensions(a, a + 1, a + 2, a + 3, a + 4),
                             ::tensor::invalid_assertion);
          break;
        case 4:
          ASSERT_THROW_DEBUG(P.get_dimensions(a, a + 1, a + 2, a + 3),
                             ::tensor::invalid_assertion);
          break;
        case 3:
          ASSERT_THROW_DEBUG(P.get_dimensions(a, a + 1, a + 2),
                             ::tensor::invalid_assertion);
          break;
        case 2:
          ASSERT_THROW_DEBUG(P.get_dimensions(a, a + 1),
                             ::tensor::invalid_assertion);
          break;
        case 1:
          ASSERT_THROW_DEBUG(P.get_dimensions(a), ::tensor::invalid_assertion);
          break;
      }
    }
  }
#endif
}

// Verify that the constructor that reshapes produces the same data
// with compatible dimensions. Here SAME means shared reference.
template <typename elt_t>
void test_constructor_with_dims(Tensor<elt_t> &P) {
  Indices d = P.dimensions();
  {
    Tensor<elt_t> P2(d);
    EXPECT_EQ(P.rank(), P2.rank());
    EXPECT_EQ(P.size(), P2.size());
    EXPECT_TRUE(d == P2.dimensions());
  }
  {
    Tensor<elt_t> P2;
    // Test constructors for tensors of given rank
    switch (P.rank()) {
      case 6: {
        P2 = Tensor<elt_t>(d[0], d[1], d[2], d[3], d[4], d[5]);
        break;
      }
      case 5: {
        P2 = Tensor<elt_t>(d[0], d[1], d[2], d[3], d[4]);
        break;
      }
      case 4: {
        P2 = Tensor<elt_t>(d[0], d[1], d[2], d[3]);
        break;
      }
      case 3: {
        P2 = Tensor<elt_t>(d[0], d[1], d[2]);
        break;
      }
      case 2: {
        P2 = Tensor<elt_t>(d[0], d[1]);
        break;
      }
      case 1: {
        P2 = Tensor<elt_t>(d[0]);
        break;
      }
      case 0: {
        P2 = Tensor<elt_t>();
        break;
      }
    }
    EXPECT_EQ(P.rank(), P2.rank());
    EXPECT_EQ(P.size(), P2.size());
    EXPECT_TRUE(d == P2.dimensions());
  }
}

// Verify that we can create a copy of a tensor with different
// dimensions, but having the same data.
template <typename elt_t>
void test_copy_constructor_with_dims(Tensor<elt_t> &P) {
  Indices d = P.dimensions();
  for (int i = 0; i < P.rank(); i++) {
    Indices copy(i + 1);
    for (int j = 0; j < P.rank(); j++) {
      if (j > i) {
        copy.at(i) *= d[j];
      } else {
        copy.at(j) = d[j];
      }
    }
    Tensor<elt_t> P2(copy, P);
    EXPECT_EQ(i + 1, P2.rank());
    EXPECT_EQ(P.size(), P2.size());
    EXPECT_TRUE(all_equal(copy, P2.dimensions()));
    unchanged(P2, P);
  }
}

// Does the same as reshape(tensor, dims), but calling explicitely the
// functions that take integers as arguments
template <typename elt_t>
Tensor<elt_t> alternate_reshape(const Tensor<elt_t> &P, const Indices &d) {
  switch (d.size()) {
    case 0:
      return P;
    case 1:
      return reshape(P, d[0]);
    case 2:
      return reshape(P, d[0], d[1]);
    case 3:
      return reshape(P, d[0], d[1], d[2]);
    case 4:
      return reshape(P, d[0], d[1], d[2], d[3]);
    case 5:
      return reshape(P, d[0], d[1], d[2], d[3], d[4]);
    case 6:
      return reshape(P, d[0], d[1], d[2], d[3], d[4], d[5]);
    default:
      return reshape(P, d);
  }
}

// Test that reshape produces a tensor with new, compatible dimensions,
// but with the same data.
template <typename elt_t>
void test_reshape(Tensor<elt_t> &P) {
  for (DimensionsProducer dp(P.dimensions()); dp; ++dp) {
    Indices new_dimensions = *dp;
    {
      Tensor<elt_t> P2 = reshape(P, new_dimensions);
      unchanged(P2, P);
      EXPECT_TRUE(all_equal(P2.dimensions(), new_dimensions));
    }
    {
      Tensor<elt_t> P2 = alternate_reshape(P, new_dimensions);
      unchanged(P2, P);
      EXPECT_TRUE(all_equal(P2.dimensions(), new_dimensions));
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RTensorTest, RTensorEmptyConstructor) { test_empty_constructor<double>(); }

TEST(RTensorTest, RTensorCopyConstructor) {
  test_over_tensors(test_copy_constructor<double>);
}

TEST(RTensorTest, RTensorDims) { test_over_tensors(test_dims<double>); }

TEST(RTensorTest, RTensorGetDimensionsErrors) {
  test_get_dimensions_errors<double>();
}

TEST(RTensorTest, RTensorCopyConstructorWithDims) {
  test_over_tensors(test_copy_constructor_with_dims<double>);
}

TEST(RTensorTest, RTensorReshape) { test_over_tensors(test_reshape<double>); }

TEST(RTensorTest, RTensorChangeDimension) {
  {
    RTensor A = RTensor::zeros(3, 2);
    RTensor B = change_dimension(A, 1, 10);
    RTensor C = RTensor::zeros(3, 10);
    EXPECT_CEQ(B, C);
  }
  {
    RTensor A = RTensor::ones(3, 1);
    RTensor B = change_dimension(A, 1, 10);
    RTensor C = RTensor::zeros(3, 10);
    C.at(0, 0) = C.at(1, 0) = C.at(2, 0) = A.at(0, 0);
    EXPECT_CEQ(B, C);
  }
}

TEST(RTensorTest, RTensorRealImag) {
  {
    RTensor A = RTensor::random(4, 3, 5);
    RTensor B = real(A);
    RTensor C = imag(A);
    ASSERT_TRUE(all_equal(B, A));
    ASSERT_TRUE(all_equal(C, 0.0));
  }
  {
    RTensor A{};
    RTensor B = imag(A);
    RTensor C = imag(A);
    ASSERT_TRUE(all_equal(B, RTensor()));
    ASSERT_TRUE(all_equal(C, RTensor()));
  }
}

TEST(RTensorTest, AccessByIndicesObject) {
  {
    RTensor A1d{1.0, 2.0};
    ASSERT_EQ(A1d.element_at(Indices{0}), 1.0);
    ASSERT_EQ(A1d.element_at(Indices{1}), 2.0);
    ASSERT_THROW_DEBUG(A1d.element_at(Indices{2}), out_of_bounds_index);
  }
  {
    RTensor A2d{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    ASSERT_EQ(A2d.element_at(Indices{0, 0}), 1.0);
    ASSERT_EQ(A2d.element_at(Indices{0, 1}), 2.0);
    ASSERT_EQ(A2d.element_at(Indices{0, 2}), 3.0);
    ASSERT_EQ(A2d.element_at(Indices{1, 0}), 4.0);
    ASSERT_EQ(A2d.element_at(Indices{1, 1}), 5.0);
    ASSERT_EQ(A2d.element_at(Indices{1, 2}), 6.0);
    ASSERT_THROW_DEBUG(A2d.element_at(Indices{2, 0}), out_of_bounds_index);
  }
}

TEST(RTensorTest, AccessByNegativeIndicesObject) {
  {
    RTensor A1d{1.0, 2.0};
    ASSERT_EQ(A1d.element_at(Indices{-2}), 1.0);
    ASSERT_EQ(A1d.element_at(Indices{-1}), 2.0);
    ASSERT_THROW_DEBUG(A1d.element_at(Indices{2}), out_of_bounds_index);
  }
  {
    RTensor A2d{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    ASSERT_EQ(A2d.element_at(Indices{0, -3}), 1.0);
    ASSERT_EQ(A2d.element_at(Indices{0, -2}), 2.0);
    ASSERT_EQ(A2d.element_at(Indices{0, -1}), 3.0);
    ASSERT_EQ(A2d.element_at(Indices{-2, 0}), 1.0);
    ASSERT_EQ(A2d.element_at(Indices{-2, 1}), 2.0);
    ASSERT_EQ(A2d.element_at(Indices{-2, 2}), 3.0);
    ASSERT_EQ(A2d.element_at(Indices{1, -3}), 4.0);
    ASSERT_EQ(A2d.element_at(Indices{1, -2}), 5.0);
    ASSERT_EQ(A2d.element_at(Indices{1, -1}), 6.0);
    ASSERT_EQ(A2d.element_at(Indices{-1, 0}), 4.0);
    ASSERT_EQ(A2d.element_at(Indices{-1, 1}), 5.0);
    ASSERT_EQ(A2d.element_at(Indices{-1, 2}), 6.0);
    ASSERT_THROW_DEBUG(A2d.element_at(Indices{-4, 0}), out_of_bounds_index);
    ASSERT_THROW_DEBUG(A2d.element_at(Indices{0, -4}), out_of_bounds_index);
  }
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CTensorTest, CTensorEmptyConstructor) {
  test_empty_constructor<cdouble>();
}

TEST(CTensorTest, CTensorCopyConstructor) {
  test_over_tensors(test_copy_constructor<cdouble>);
}

TEST(CTensorTest, CTensorDims) { test_over_tensors(test_dims<cdouble>); }

TEST(CTensorTest, CTensorGetDimensionsErrors) {
  test_get_dimensions_errors<cdouble>();
}

TEST(CTensorTest, CTensorCopyConstructorWithDims) {
  test_over_tensors(test_copy_constructor_with_dims<cdouble>);
}

TEST(CTensorTest, CTensorReshape) { test_over_tensors(test_reshape<cdouble>); }

TEST(CTensorTest, CTensorChangeDimension) {
  {
    CTensor A = CTensor::zeros(3, 2);
    CTensor B = change_dimension(A, 1, 10);
    CTensor C = CTensor::zeros(3, 10);
    EXPECT_CEQ(B, C);
  }
  {
    CTensor A = CTensor::ones(3, 1);
    CTensor B = change_dimension(A, 1, 10);
    CTensor C = CTensor::zeros(3, 10);
    C.at(0, 0) = C.at(1, 0) = C.at(2, 0) = A.at(0, 0);
    EXPECT_CEQ(B, C);
  }
}

TEST(CTensorTest, CTensorRealImag) {
  {
    CTensor A = CTensor::random(4, 3, 5);
    RTensor B = real(A);
    RTensor C = imag(A);
    ASSERT_TRUE(all_equal(B.dimensions(), A.dimensions()));
    ASSERT_TRUE(all_equal(C.dimensions(), A.dimensions()));
    for (size_t i = 0; i < A.size(); ++i) {
      ASSERT_TRUE(A[i].real() == B[i]);
      ASSERT_TRUE(A[i].imag() == C[i]);
    }
  }
  {
    CTensor A{};
    RTensor B = imag(A);
    RTensor C = imag(A);
    ASSERT_TRUE(all_equal(B, CTensor()));
    ASSERT_TRUE(all_equal(C, CTensor()));
  }
}

TEST(CTensorTest, RTensorToComplex) {
  RTensor A{1.0, 0.0, -2.0};
  RTensor B{0.2, -0.1, 0.0};
  {
    const CTensor C = to_complex(A);
    EXPECT_EQ(A.dimensions(), C.dimensions());
    EXPECT_EQ(A[0], real(C[0]));
    EXPECT_EQ(A[1], real(C[1]));
    EXPECT_EQ(A[2], real(C[2]));
    EXPECT_TRUE(std::all_of(C.begin(), C.end(),
                            [](cdouble z) { return imag(z) == 0; }));
  }
  {
    const CTensor C = to_complex(A, B);
    EXPECT_EQ(A.dimensions(), C.dimensions());
    EXPECT_EQ(A[0], real(C[0]));
    EXPECT_EQ(A[1], real(C[1]));
    EXPECT_EQ(A[2], real(C[2]));
    EXPECT_EQ(B[0], imag(C[0]));
    EXPECT_EQ(B[1], imag(C[1]));
    EXPECT_EQ(B[2], imag(C[2]));
  }
  ASSERT_THROW_DEBUG(to_complex(RTensor{1.0}, RTensor{0.0, 2.0}),
                     ::tensor::dimensions_mismatch);
}
}  // namespace tensor_test
