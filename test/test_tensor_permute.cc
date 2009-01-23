// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <algorithm>
#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

  using tensor::index;

  ////////////////////////////////////////////////////////////////////////
  //
  // TESTING TENSOR UNARY OPERATIONS
  //
  template<typename elt_t>
  bool eq_permute_2(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2;
    A.get_dimensions(&a1, &a2);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        if (A(i,j) != P(j,i)) return false;
      }
    }
    return true;
  }

  //
  // THREE-DIMENSIONAL TENSORS
  //
  template<typename elt_t>
  bool eq_permute_3(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2, a3;
    A.get_dimensions(&a1, &a2, &a3);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        for (index k = 0; k < a3; k++) {
          switch (code) {
          case 123: if (A(i,j,k) != P(j,i,k)) return false; break;
          case 133: if (A(i,j,k) != P(k,j,i)) return false; break;
          case 233: if (A(i,j,k) != P(i,k,j)) return false; break;
          }
        }
      }
    }
    return true;
  }

  //
  // FOUR-DIMENSIONAL TENSORS
  //
  template<typename elt_t>
  bool eq_permute_4(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2, a3, a4;
    A.get_dimensions(&a1, &a2, &a3, &a4);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        for (index k = 0; k < a3; k++) {
          for (index l = 0; l < a4; l++) {
            switch (code) {
            case 124: if (A(i,j,k,l) != P(j,i,k,l)) return false; break;
            case 134: if (A(i,j,k,l) != P(k,j,i,l)) return false; break;
            case 144: if (A(i,j,k,l) != P(l,j,k,i)) return false; break;
            case 234: if (A(i,j,k,l) != P(i,k,j,l)) return false; break;
            case 244: if (A(i,j,k,l) != P(i,l,k,j)) return false; break;
            case 344: if (A(i,j,k,l) != P(i,j,l,k)) return false; break;
            }
          }
        }
      }
    }
    return true;
  }

  //
  // FIVE DIMENSIONAL TENSOR
  //

  template<typename elt_t>
  bool eq_permute_5(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2, a3, a4, a5;
    A.get_dimensions(&a1, &a2, &a3, &a4, &a5);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        for (index k = 0; k < a3; k++) {
          for (index l = 0; l < a4; l++) {
            for (index m = 0; m < a5; m++) {
              switch (code) {
              case 125: if (A(i,j,k,l,m) != P(j,i,k,l,m)) return false; break;
              case 135: if (A(i,j,k,l,m) != P(k,j,i,l,m)) return false; break;
              case 145: if (A(i,j,k,l,m) != P(l,j,k,i,m)) return false; break;
              case 155: if (A(i,j,k,l,m) != P(m,j,k,l,i)) return false; break;
              case 235: if (A(i,j,k,l,m) != P(i,k,j,l,m)) return false; break;
              case 245: if (A(i,j,k,l,m) != P(i,l,k,j,m)) return false; break;
              case 255: if (A(i,j,k,l,m) != P(i,m,k,l,j)) return false; break;
              case 345: if (A(i,j,k,l,m) != P(i,j,l,k,m)) return false; break;
              case 355: if (A(i,j,k,l,m) != P(i,j,m,l,k)) return false; break;
              case 455: if (A(i,j,k,l,m) != P(i,j,k,m,l)) return false; break;
              }
            }
          }
        }
      }
    }
    return true;
  }

  //
  // SIX DIMENSIONAL TENSOR
  //

  template<typename elt_t>
  bool eq_permute_6(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2, a3, a4, a5, a6;
    A.get_dimensions(&a1, &a2, &a3, &a4, &a5, &a6);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        for (index k = 0; k < a3; k++) {
          for (index l = 0; l < a4; l++) {
            for (index m = 0; m < a5; m++) {
              for (index n = 0; n < a6; n++) {
                switch (code) {
                case 126: if (A(i,j,k,l,m,n) != P(j,i,k,l,m,n)) return false; break;
                case 136: if (A(i,j,k,l,m,n) != P(k,j,i,l,m,n)) return false; break;
                case 146: if (A(i,j,k,l,m,n) != P(l,j,k,i,m,n)) return false; break;
                case 156: if (A(i,j,k,l,m,n) != P(m,j,k,l,i,n)) return false; break;
                case 166: if (A(i,j,k,l,m,n) != P(n,j,k,l,m,i)) return false; break;
                case 236: if (A(i,j,k,l,m,n) != P(i,k,j,l,m,n)) return false; break;
                case 246: if (A(i,j,k,l,m,n) != P(i,l,k,j,m,n)) return false; break;
                case 256: if (A(i,j,k,l,m,n) != P(i,m,k,l,j,n)) return false; break;
                case 266: if (A(i,j,k,l,m,n) != P(i,n,k,l,m,j)) return false; break;
                case 346: if (A(i,j,k,l,m,n) != P(i,j,l,k,m,n)) return false; break;
                case 356: if (A(i,j,k,l,m,n) != P(i,j,m,l,k,n)) return false; break;
                case 366: if (A(i,j,k,l,m,n) != P(i,j,n,l,m,k)) return false; break;
                case 456: if (A(i,j,k,l,m,n) != P(i,j,k,m,l,n)) return false; break;
                case 466: if (A(i,j,k,l,m,n) != P(i,j,k,n,m,l)) return false; break;
                case 566: if (A(i,j,k,l,m,n) != P(i,j,k,l,n,m)) return false; break;
                }
              }
            }
          }
        }
      }
    }
    return true;
  }


  template<typename elt_t>
  void test_permute(Tensor<elt_t> &A)
  {
    const Indices &dA = A.dimensions();
    for (int i = 0; i < A.rank(); i++) {
      for (int j = 0; j < A.rank(); j++) {
        Tensor<elt_t> P = permute(A, i, j);
        Indices dP = P.dimensions();
        std::swap(dP.at(i), dP.at(j));
        if (dP.size() != dA.size())
          abort();
        EXPECT_EQ(dP.size(), dA.size());
        EXPECT_TRUE(std::equal(dP.begin(), dP.end(), dA.begin()));
        if (i == j) {
          EXPECT_EQ(A, P);
        } else {
          int code = (i < j)?
                     (i+1)*100 + (j+1)*10 + A.rank() :
                     (j+1)*100 + (i+1)*10 + A.rank();
          switch (A.rank()) {
          case 2: EXPECT_TRUE(eq_permute_2(A, P, code)); break;
          case 3: EXPECT_TRUE(eq_permute_3(A, P, code)); break;
          case 4: EXPECT_TRUE(eq_permute_4(A, P, code)); break;
          case 5: EXPECT_TRUE(eq_permute_5(A, P, code)); break;
          case 6: EXPECT_TRUE(eq_permute_6(A, P, code)); break;
          }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(TensorPermuteTest, RTensorPermute) {
    test_over_all_tensors<double>(test_permute<double>, 6, 5);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(TensorPermuteTest, CTensorPermute) {
    test_over_all_tensors<cdouble>(test_permute<cdouble>, 6, 5);
  }

} // namespace tensor_test

