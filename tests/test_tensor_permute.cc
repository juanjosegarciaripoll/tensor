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

#include <algorithm>
#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>

namespace tensor_test {

  using tensor::index;

  ////////////////////////////////////////////////////////////////////////
  //
  // TESTING TENSOR UNARY OPERATIONS
  //
  void error_wrong_code(int code)
  {
    std::cerr << "Not a valid permutation code: " << code;
    abort();
  }

  template<typename elt_t>
  bool eq_permute_2(const Tensor<elt_t> &A, const Tensor<elt_t> &P, int code)
  {
    index a1, a2;
    A.get_dimensions(&a1, &a2);
    for (index i = 0; i < a1; i++) {
      for (index j = 0; j < a2; j++) {
        switch (code) {
        case 122:
        case 212: if (A(i,j) != P(j,i)) return false; break;
        default:  error_wrong_code(code);
        }
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
          default:  error_wrong_code(code);
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
            default:  error_wrong_code(code);
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
              default:  error_wrong_code(code);
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
                default:  error_wrong_code(code);
                }
              }
            }
          }
        }
      }
    }
    return true;
  }

  template<typename elt_t, int i, int j>
  void test_permute(Tensor<elt_t> &A)
  {
    const Indices &dA = A.dimensions();
    {
        Tensor<elt_t> P = permute(A, i, j);
        Indices dP = P.dimensions();
        std::swap(dP.at(i), dP.at(j));
        if (dP.size() != dA.size())
          abort();
        EXPECT_EQ(dP.size(), dA.size());
        EXPECT_TRUE(std::equal(dP.begin(), dP.end(), dA.begin()));
        if (i == j) {
          EXPECT_TRUE(all_equal(A, P));
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

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

#define RTENSOR_TEST(i,j,r)                                             \
  TEST(TensorPermuteTest, RTensorPermute ## i ## j ## r) {              \
    test_over_fixed_rank_tensors<double>(test_permute<double,i,j>, r, 5); \
  }

  RTENSOR_TEST(0,0,1)


  RTENSOR_TEST(0,0,2)
  RTENSOR_TEST(0,1,2)

  RTENSOR_TEST(1,0,2)
  RTENSOR_TEST(1,1,2)


  RTENSOR_TEST(0,0,3)
  RTENSOR_TEST(0,1,3)
  RTENSOR_TEST(0,2,3)

  RTENSOR_TEST(1,0,3)
  RTENSOR_TEST(1,1,3)
  RTENSOR_TEST(1,2,3)

  RTENSOR_TEST(2,0,3)
  RTENSOR_TEST(2,1,3)
  RTENSOR_TEST(2,2,3)


  RTENSOR_TEST(0,0,4)
  RTENSOR_TEST(0,1,4)
  RTENSOR_TEST(0,2,4)
  RTENSOR_TEST(0,3,4)

  RTENSOR_TEST(1,0,4)
  RTENSOR_TEST(1,1,4)
  RTENSOR_TEST(1,2,4)
  RTENSOR_TEST(1,3,4)

  RTENSOR_TEST(2,0,4)
  RTENSOR_TEST(2,1,4)
  RTENSOR_TEST(2,2,4)
  RTENSOR_TEST(2,3,4)

  RTENSOR_TEST(3,0,4)
  RTENSOR_TEST(3,1,4)
  RTENSOR_TEST(3,2,4)
  RTENSOR_TEST(3,3,4)


  RTENSOR_TEST(0,0,5)
  RTENSOR_TEST(0,1,5)
  RTENSOR_TEST(0,2,5)
  RTENSOR_TEST(0,3,5)
  RTENSOR_TEST(0,4,5)

  RTENSOR_TEST(1,0,5)
  RTENSOR_TEST(1,1,5)
  RTENSOR_TEST(1,2,5)
  RTENSOR_TEST(1,3,5)
  RTENSOR_TEST(1,4,5)

  RTENSOR_TEST(2,0,5)
  RTENSOR_TEST(2,1,5)
  RTENSOR_TEST(2,2,5)
  RTENSOR_TEST(2,3,5)
  RTENSOR_TEST(2,4,5)

  RTENSOR_TEST(3,0,5)
  RTENSOR_TEST(3,1,5)
  RTENSOR_TEST(3,2,5)
  RTENSOR_TEST(3,3,5)
  RTENSOR_TEST(3,4,5)

  RTENSOR_TEST(4,0,5)
  RTENSOR_TEST(4,1,5)
  RTENSOR_TEST(4,2,5)
  RTENSOR_TEST(4,3,5)
  RTENSOR_TEST(4,4,5)


  RTENSOR_TEST(0,0,6)
  RTENSOR_TEST(0,1,6)
  RTENSOR_TEST(0,2,6)
  RTENSOR_TEST(0,3,6)
  RTENSOR_TEST(0,4,6)
  RTENSOR_TEST(0,5,6)

  RTENSOR_TEST(1,0,6)
  RTENSOR_TEST(1,1,6)
  RTENSOR_TEST(1,2,6)
  RTENSOR_TEST(1,3,6)
  RTENSOR_TEST(1,4,6)
  RTENSOR_TEST(1,5,6)

  RTENSOR_TEST(2,0,6)
  RTENSOR_TEST(2,1,6)
  RTENSOR_TEST(2,2,6)
  RTENSOR_TEST(2,3,6)
  RTENSOR_TEST(2,4,6)
  RTENSOR_TEST(2,5,6)

  RTENSOR_TEST(3,0,6)
  RTENSOR_TEST(3,1,6)
  RTENSOR_TEST(3,2,6)
  RTENSOR_TEST(3,3,6)
  RTENSOR_TEST(3,4,6)
  RTENSOR_TEST(3,5,6)

  RTENSOR_TEST(4,0,6)
  RTENSOR_TEST(4,1,6)
  RTENSOR_TEST(4,2,6)
  RTENSOR_TEST(4,3,6)
  RTENSOR_TEST(4,4,6)
  RTENSOR_TEST(4,5,6)

  RTENSOR_TEST(5,0,6)
  RTENSOR_TEST(5,1,6)
  RTENSOR_TEST(5,2,6)
  RTENSOR_TEST(5,3,6)
  RTENSOR_TEST(5,4,6)
  RTENSOR_TEST(5,5,6)

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //


#define CTENSOR_TEST(i,j,r)                                             \
  TEST(TensorPermuteTest, CTensorPermute ## i ## j ## r) {              \
    test_over_fixed_rank_tensors<cdouble>(test_permute<cdouble,i,j>, r, 5); \
  }

  CTENSOR_TEST(0,0,1)


  CTENSOR_TEST(0,0,2)
  CTENSOR_TEST(0,1,2)

  CTENSOR_TEST(1,0,2)
  CTENSOR_TEST(1,1,2)


  CTENSOR_TEST(0,0,3)
  CTENSOR_TEST(0,1,3)
  CTENSOR_TEST(0,2,3)

  CTENSOR_TEST(1,0,3)
  CTENSOR_TEST(1,1,3)
  CTENSOR_TEST(1,2,3)

  CTENSOR_TEST(2,0,3)
  CTENSOR_TEST(2,1,3)
  CTENSOR_TEST(2,2,3)


  CTENSOR_TEST(0,0,4)
  CTENSOR_TEST(0,1,4)
  CTENSOR_TEST(0,2,4)
  CTENSOR_TEST(0,3,4)

  CTENSOR_TEST(1,0,4)
  CTENSOR_TEST(1,1,4)
  CTENSOR_TEST(1,2,4)
  CTENSOR_TEST(1,3,4)

  CTENSOR_TEST(2,0,4)
  CTENSOR_TEST(2,1,4)
  CTENSOR_TEST(2,2,4)
  CTENSOR_TEST(2,3,4)

  CTENSOR_TEST(3,0,4)
  CTENSOR_TEST(3,1,4)
  CTENSOR_TEST(3,2,4)
  CTENSOR_TEST(3,3,4)


  CTENSOR_TEST(0,0,5)
  CTENSOR_TEST(0,1,5)
  CTENSOR_TEST(0,2,5)
  CTENSOR_TEST(0,3,5)
  CTENSOR_TEST(0,4,5)

  CTENSOR_TEST(1,0,5)
  CTENSOR_TEST(1,1,5)
  CTENSOR_TEST(1,2,5)
  CTENSOR_TEST(1,3,5)
  CTENSOR_TEST(1,4,5)

  CTENSOR_TEST(2,0,5)
  CTENSOR_TEST(2,1,5)
  CTENSOR_TEST(2,2,5)
  CTENSOR_TEST(2,3,5)
  CTENSOR_TEST(2,4,5)

  CTENSOR_TEST(3,0,5)
  CTENSOR_TEST(3,1,5)
  CTENSOR_TEST(3,2,5)
  CTENSOR_TEST(3,3,5)
  CTENSOR_TEST(3,4,5)

  CTENSOR_TEST(4,0,5)
  CTENSOR_TEST(4,1,5)
  CTENSOR_TEST(4,2,5)
  CTENSOR_TEST(4,3,5)
  CTENSOR_TEST(4,4,5)


  CTENSOR_TEST(0,0,6)
  CTENSOR_TEST(0,1,6)
  CTENSOR_TEST(0,2,6)
  CTENSOR_TEST(0,3,6)
  CTENSOR_TEST(0,4,6)
  CTENSOR_TEST(0,5,6)

  CTENSOR_TEST(1,0,6)
  CTENSOR_TEST(1,1,6)
  CTENSOR_TEST(1,2,6)
  CTENSOR_TEST(1,3,6)
  CTENSOR_TEST(1,4,6)
  CTENSOR_TEST(1,5,6)

  CTENSOR_TEST(2,0,6)
  CTENSOR_TEST(2,1,6)
  CTENSOR_TEST(2,2,6)
  CTENSOR_TEST(2,3,6)
  CTENSOR_TEST(2,4,6)
  CTENSOR_TEST(2,5,6)

  CTENSOR_TEST(3,0,6)
  CTENSOR_TEST(3,1,6)
  CTENSOR_TEST(3,2,6)
  CTENSOR_TEST(3,3,6)
  CTENSOR_TEST(3,4,6)
  CTENSOR_TEST(3,5,6)

  CTENSOR_TEST(4,0,6)
  CTENSOR_TEST(4,1,6)
  CTENSOR_TEST(4,2,6)
  CTENSOR_TEST(4,3,6)
  CTENSOR_TEST(4,4,6)
  CTENSOR_TEST(4,5,6)

  CTENSOR_TEST(5,0,6)
  CTENSOR_TEST(5,1,6)
  CTENSOR_TEST(5,2,6)
  CTENSOR_TEST(5,3,6)
  CTENSOR_TEST(5,4,6)
  CTENSOR_TEST(5,5,6)

} // namespace tensor_test

