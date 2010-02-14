// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

  using namespace tensor;

  IndexRange *range2(tensor::index i0, tensor::index i2, tensor::index i1)
  {
    tensor::index l = (i2 - i0) / i1 + 1;
    Indices output(l);
    for (int i = 0; i < l; i++, i0 += i1) {
      output.at(i) = i0;
    }
    return new IndexRange(output);
  }

  template<typename elt_t> void test_range1(Tensor<elt_t> &P) {
    Tensor<elt_t> Paux = P;

    for (tensor::index i = 0; i < P.dimension(0); i++) {
      Tensor<elt_t> t = P(range(i));
      EXPECT_EQ(P.rank(), t.rank());
      EXPECT_EQ(1, t.size());
      EXPECT_EQ(t[0], P(i));
      unchanged(P, Paux);

      Tensor<elt_t> t3 = P(range(i,i));
      EXPECT_EQ(t3, t);
      unchanged(P, Paux);

      Tensor<elt_t> t4 = P(range(i,i,1));
      EXPECT_EQ(t4, t);
      unchanged(P, Paux);

      if (i+1 < P.dimension(0)) {
        Tensor<elt_t> t5 = P(range(i,i+1,2));
        EXPECT_EQ(t5, t);
        unchanged(P, Paux);
      }

      Indices ndx(1);
      ndx.at(0) = i;
      Tensor<elt_t> t6 = P(range(ndx));
      EXPECT_EQ(t6, t);
      unchanged(P, Paux);
    }
    Tensor<elt_t> t = P(range());
    EXPECT_EQ(P, t);
    unchanged(P, Paux);
  }

  template<typename elt_t> Tensor<elt_t>
  slow_range2(const Tensor<elt_t> &P,
              tensor::index i0, tensor::index i2, tensor::index i1,
              tensor::index j0, tensor::index j2, tensor::index j1)
  {
    Indices i(2);
    i.at(0) = (i2 - i0) / i1 + 1;
    i.at(1) = (j2 - j0) / j1 + 1;
    Tensor<elt_t> t(i);
    for (tensor::index i = i0, x = 0; i <= i2; i += i1, x++) {
      for (tensor::index j = j0, y = 0; j <= j2; j += j1, y++) {
        t.at(x,y) = P(i,j);
      }
    }
    return t;
  }

  template<typename elt_t> Tensor<elt_t>
  slow_range3(const Tensor<elt_t> &P,
              tensor::index i0, tensor::index i2, tensor::index i1,
              tensor::index j0, tensor::index j2, tensor::index j1,
              tensor::index k0, tensor::index k2, tensor::index k1)
  {
    Indices i(3);
    i.at(0) = (i2 - i0) / i1 + 1;
    i.at(1) = (j2 - j0) / j1 + 1;
    i.at(2) = (k2 - k0) / k1 + 1;
    Tensor<elt_t> t(i);
    for (tensor::index i = i0, x = 0; i <= i2; i += i1, x++) {
      for (tensor::index j = j0, y = 0; j <= j2; j += j1, y++) {
        for (tensor::index k = k0, z = 0; k <= k2; k += k1, z++) {
          t.at(x,y,z) = P(i,j,k);
        }
      }
    }
    return t;
  }

  template<typename elt_t> void test_range2(Tensor<elt_t> &P) {
    tensor::index rows = P.dimension(0);
    tensor::index cols = P.dimension(1);
    Tensor<elt_t> Paux = P;

    for (tensor::index i = 0; i < rows; i++) {
      for (tensor::index j = 0; j < cols; j++) {
        Tensor<elt_t> t = P(range(i), range(j));
        EXPECT_EQ(P.rank(), t.rank());
        EXPECT_EQ(1, t.size());
        EXPECT_EQ(t[0], P(i,j));
        Tensor<elt_t> t2 = P(range(i), range(j,j));
        EXPECT_EQ(t2, t);
        Tensor<elt_t> t3 = P(range(i,i), range(j,j));
        EXPECT_EQ(t3, t);
        Tensor<elt_t> t4 = P(range(i,i), range(j));
        EXPECT_EQ(t4, t);
        if (i+1 < P.dimension(0)) {
          Tensor<elt_t> t5 = P(range(i,i+1,2), range(j));
          EXPECT_EQ(t5, t);
        }
        Tensor<elt_t> t6 = P(range2(i,i+1,2), range(j));
        EXPECT_EQ(t6, t);
        Tensor<elt_t> t7 = P(range(i), range(j,j));
        EXPECT_EQ(t7, t);
      }
    }
    unchanged(P, Paux);

    for (tensor::index i1 = 1; i1 < 4; i1++) {
      for (tensor::index j1 = 1; j1 < 4; j1++) {
        for (tensor::index i0 = 0; i0 < rows; i0++) {
          for (tensor::index j0 = 0; j0 < cols; j0++) {
            for (tensor::index i2 = i0; i2 < rows; i2++) {
              for (tensor::index j2 = j0; j2 < cols; j2++) {
                Tensor<elt_t> t1 = slow_range2(P, i0,i2,i1,j0,j2,j1);
                Tensor<elt_t> t2 = P(range(i0,i2,i1), range(j0,j2,j1));
                EXPECT_EQ(t2, t1);
                Tensor<elt_t> t3 = P(range2(i0,i2,i1), range(j0,j2,j1));
                EXPECT_EQ(t3, t1);
                Tensor<elt_t> t4 = P(range(i0,i2,i1), range2(j0,j2,j1));
                EXPECT_EQ(t4, t1);
                if (t1.dimension(0) == 1) {
                  Tensor<elt_t> t5 = P(range(i0), range(j0,j2,j1));
                  EXPECT_EQ(t5, t1);
                }
                if (t1.dimension(1) == 1) {
                  Tensor<elt_t> t6 = P(range(i0,i2,i1), range(j0));
                  EXPECT_EQ(t6, t1);
                }
                if (i1 == 1 && t1.dimension(0) == P.dimension(0)) {
                  Tensor<elt_t> t7 = P(range(), range(j0,j2,j1));
                  EXPECT_EQ(t7, t1);
                }
                if (j1 == 1 && t1.dimension(1) == P.dimension(1)) {
                  Tensor<elt_t> t7 = P(range(i0,i2,i1), range());
                  EXPECT_EQ(t7, t1);
                }
              }
            }
          }
        }
      }
    }
    unchanged(P, Paux);

    Tensor<elt_t> t = P(range(), range());
    EXPECT_EQ(P, t);
    unchanged(P, Paux);
  }


  template<typename elt_t> void test_range3(Tensor<elt_t> &P) {
    tensor::index d0 = P.dimension(0);
    tensor::index d1 = P.dimension(1);
    tensor::index d2 = P.dimension(2);
    Tensor<elt_t> Paux = P;

    for (tensor::index i = 0; i < d0; i++) {
      for (tensor::index j = 0; j < d1; j++) {
        for (tensor::index k = 0; k < d2; k++) {
          Tensor<elt_t> t = P(range(i), range(j), range(k));
          EXPECT_EQ(P.rank(), t.rank());
          EXPECT_EQ(1, t.size());
          EXPECT_EQ(t[0], P(i,j,k));

          Tensor<elt_t> t2 = P(range(i), range(j,j), range(k));
          EXPECT_EQ(t2, t);
          Tensor<elt_t> t3 = P(range(i,i), range(j,j), range(k));
          EXPECT_EQ(t3, t);
          Tensor<elt_t> t4 = P(range(i,i), range(j), range(k));
          EXPECT_EQ(t4, t);
          if (i+1 < P.dimension(0)) {
            Tensor<elt_t> t5 = P(range(i,i+1,2), range(j), range(k));
            EXPECT_EQ(t5, t);
          }
          Tensor<elt_t> t6 = P(range2(i,i+1,2), range(j), range(k));
          EXPECT_EQ(t6, t);
          Tensor<elt_t> t7 = P(range(i), range(j,j), range(k));
          EXPECT_EQ(t7, t);
          Tensor<elt_t> t8 = P(range(i), range(j,j), range(k,k));
          EXPECT_EQ(t8, t);
          Tensor<elt_t> t9 = P(range(i), range(j,j), range2(k,k+1,2));
          EXPECT_EQ(t9, t);
        }
      }
    }
    unchanged(P, Paux);

    for (tensor::index i1 = 1; i1 < 3; i1++) {
      for (tensor::index j1 = 1; j1 < 3; j1++) {
        for (tensor::index k1 = 1; k1 < 3; k1++) {
          for (tensor::index i0 = 0; i0 < d0; i0++) {
            for (tensor::index j0 = 0; j0 < d1; j0++) {
              for (tensor::index k0 = 0; k0 < d2; k0++) {
                for (tensor::index i2 = i0; i2 < d0; i2++) {
                  for (tensor::index j2 = j0; j2 < d1; j2++) {
                    for (tensor::index k2 = k0; k2 < d2; k2++) {
                      Tensor<elt_t> t1 = slow_range3(P, i0,i2,i1,j0,j2,j1,k0,k2,k1);
                      Tensor<elt_t> t2 =
                        P(range(i0,i2,i1), range(j0,j2,j1), range(k0,k2,k1));
                      EXPECT_EQ(t2, t1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    unchanged(P, Paux);

    Tensor<elt_t> t = P(range(), range(), range());
    EXPECT_EQ(P, t);
    unchanged(P, Paux);
  }

  template<typename elt_t> void test_range(Tensor<elt_t> &P) {
    switch (P.rank()) {
    case 1: test_range1(P); break;
    case 2: test_range2(P); break;
    case 3: test_range3(P); break;
    }
  }

  /////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(SliceTest, SliceRTensorTest) {
    //test_over_tensors<double>(test_range<double>,2,10,100);
    test_over_all_tensors<double>(test_range<double>,3);
  }

  /////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(SliceTest, SliceCTensorTest) {
    //test_over_tensors<cdouble>(test_range<cdouble>,2,10,100);
    test_over_all_tensors<cdouble>(test_range<cdouble>,3,6);
  }

} // namespace tensor_test
