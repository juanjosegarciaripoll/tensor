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

#include <tensor/sparse.h>
#include "loops.h"


#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>

namespace tensor_test {

  //
  // EMPTY SPARSE MATRICES
  //
  template<typename elt_t>
  void test_empty_constructor() {
    {
    SCOPED_TRACE("0x0");
    Sparse<elt_t> S;
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_TRUE(all_equal(igen << 0, S.priv_row_start()));
    //EXPECT_EQ(S.priv_row_start(), (igen << 0));
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>(0,0)));
    }
    {
    SCOPED_TRACE("0x1");
    Sparse<elt_t> S(0,2);
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_TRUE(all_equal(igen << 0, S.priv_row_start()));
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>::zeros(0,2)));
    }
    {
    SCOPED_TRACE("2x0");
    Sparse<elt_t> S(2,0);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 0 << 0, S.priv_row_start()));
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>::zeros(2,0)));
    }
  }

  TEST(RSparseTest, RSparseEmptyConstructor) {
    test_empty_constructor<double>();
  }

  TEST(CSparseTest, CSparseEmptyConstructor) {
    test_empty_constructor<cdouble>();
  }

  //
  // SMALL SPARSE MATRICES
  //
  template<typename elt_t>
  void test_small_constructor() {
    {
    // sparse([1]);
    SCOPED_TRACE("1x1");
    Tensor<elt_t> T(igen << 1 << 1, 
                    gen<elt_t>(1.0));
    Sparse<elt_t> S(T);
    EXPECT_EQ(1, S.rows());
    EXPECT_EQ(1, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0), S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), T));
    }
    {
    // sparse([1, 0; 0, 2]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 2, 
                    gen<elt_t>(1.0) << 0.0 << 0.0 << 2.0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1 << 2, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0 << 1, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0) << 2.0, S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), T));
    }
    {
    // sparse([1, 2; 0, 3]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 2, 
                    gen<elt_t>(1.0) << 0.0 << 2.0 << 3.0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 2 << 3, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0 << 1 << 1, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0) << 2.0 << 3.0, S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), T));
    }
    {
    // sparse([1, 0; 2, 3]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 2, 
                    gen<elt_t>(1.0) << 2.0 << 0.0 << 3.0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1 << 3, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0 << 0 << 1, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0) << 2.0 << 3.0, S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), T));
    }
    {
    // sparse([1, 0, 4; 2, 3, 0]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 3, 
                    gen<elt_t>(1.0) << 2 << 0 << 3 << 4 << 0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(3, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 2 << 4, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0 << 2 << 0 << 1, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0) << 4.0 << 2.0 << 3.0, S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), T));
    }
  }

  TEST(RSparseTest, RSparseSmallConstructor) {
    test_small_constructor<double>();
  }

  TEST(CSparseTest, CSparseSmallConstructor) {
    test_small_constructor<cdouble>();
  }

  //
  // SPARSE IDENTITITES BUILT BY HAND
  //
  template<typename elt_t>
  void test_sparse_eye_small() {
    {
    // speye(1,1)
    SCOPED_TRACE("1x1");
    Sparse<elt_t> S = Sparse<elt_t>::eye(1,1);
    EXPECT_EQ(1, S.rows());
    EXPECT_EQ(1, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0), S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>::eye(1,1)));
    }
    {
    // speye(2,1)
    SCOPED_TRACE("2x1");
    Sparse<elt_t> S = Sparse<elt_t>::eye(2,1);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(1, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1 << 1, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0), S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>::eye(2,1)));
    }
    {
    // speye(1,2)
    SCOPED_TRACE("1x2");
    Sparse<elt_t> S = Sparse<elt_t>::eye(1,2);
    EXPECT_EQ(1, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 1, S.priv_row_start()));
    EXPECT_TRUE(all_equal(igen << 0, S.priv_column()));
    EXPECT_TRUE(all_equal(gen<elt_t>(1.0), S.priv_data()));
    EXPECT_TRUE(all_equal(full(S), Tensor<elt_t>::eye(1,2)));
    }
  }

  TEST(RSparseTest, RSparseEyeSmall) {
    test_sparse_eye_small<double>();
  }

  TEST(CSparseTest, CSparseEyeSmall) {
    test_sparse_eye_small<cdouble>();
  }

  //
  // SPARSE IDENTITITES ARBITRARY SIZES
  //
  template<typename elt_t>
  void test_sparse_eye(Tensor<elt_t> &t) {
    tensor::index rows = t.rows(), cols = t.columns(), k = std::min(rows, cols);
    Tensor<elt_t> taux = Tensor<elt_t>::eye(rows, cols);
    Sparse<elt_t> saux = Sparse<elt_t>::eye(rows, cols);

    EXPECT_TRUE(all_equal(Indices::range(0,k-1), saux.priv_column()));
    Vector<elt_t> v(k);
    std::fill(v.begin(), v.end(), number_one<elt_t>());
    EXPECT_TRUE(all_equal(v, saux.priv_data()));
    EXPECT_TRUE(all_equal(taux, full(saux)));
    EXPECT_TRUE(all_equal(Sparse<elt_t>(taux), saux));
    EXPECT_EQ(rows, saux.rows());
    EXPECT_EQ(cols, saux.columns());
    EXPECT_EQ(rows+1, saux.priv_row_start().size());
    for (tensor::index i = 0; i <= rows; i++) {
      EXPECT_EQ(saux.priv_row_start()[i], std::min(i, k));
    }
  }

  TEST(RSparseTest, RSparseEye) {
    test_over_fixed_rank_tensors<double>(test_sparse_eye<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseEye) {
    test_over_fixed_rank_tensors<cdouble>(test_sparse_eye<cdouble>, 2, 7);
  }

  //
  // SPARSE IDENTITITES BUILT BY HAND
  //
  template<typename elt_t>
  void test_sparse_random_small() {
    {
    // sprandom(0,0)
    SCOPED_TRACE("0x0");
    Sparse<elt_t> S = Sparse<elt_t>::random(0,0);
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_TRUE(all_equal(igen << 0, S.priv_row_start()));
    EXPECT_TRUE(all_equal(Indices(), S.priv_column()));
    EXPECT_TRUE(all_equal(Vector<elt_t>(), S.priv_data()));
    EXPECT_TRUE(all_equal(Tensor<elt_t>(0,0), full(S)));
    }
    {
    // sprandom(0,1)
    SCOPED_TRACE("0x1");
    Sparse<elt_t> S = Sparse<elt_t>::random(0,1);
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(1, S.columns());
    EXPECT_TRUE(all_equal(igen << 0, S.priv_row_start()));
    EXPECT_TRUE(all_equal(Indices(), S.priv_column()));
    EXPECT_TRUE(all_equal(Vector<elt_t>(), S.priv_data()));
    EXPECT_TRUE(all_equal(Tensor<elt_t>(0,1), full(S)));
    }
    {
    // sprandom(2,0)
    SCOPED_TRACE("2x0");
    Sparse<elt_t> S = Sparse<elt_t>::random(2,0);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_TRUE(all_equal(igen << 0 << 0 << 0, S.priv_row_start()));
    EXPECT_TRUE(all_equal(Indices(), S.priv_column()));
    EXPECT_TRUE(all_equal(Vector<elt_t>(), S.priv_data()));
    EXPECT_TRUE(all_equal(Tensor<elt_t>(2,0), full(S)));
    }
  }

  TEST(RSparseTest, RSparseRandomSmall) {
    test_sparse_random_small<double>();
  }

  TEST(CSparseTest, CSparseRandomSmall) {
    test_sparse_random_small<cdouble>();
  }

  //
  // SPARSE RANDOM MATRICES ARBITRARY SIZES
  //
  template<typename elt_t>
  void test_sparse_random(Tensor<elt_t> &t) {
    tensor::index rows = t.rows(), cols = t.columns();
    {
    Sparse<elt_t> s = Sparse<elt_t>::random(rows, cols);
    EXPECT_EQ(rows, s.rows());
    EXPECT_EQ(cols, s.columns());
    Tensor<elt_t> t = full(s);
    EXPECT_TRUE(all_equal(Sparse<elt_t>(t), s));
    tensor::index zero = std::count(t.begin(), t.end(), number_zero<elt_t>());
    tensor::index nonzero = t.size() - zero;
    EXPECT_EQ(nonzero, s.length());
    EXPECT_EQ(nonzero, s.priv_data().size());
    EXPECT_EQ(nonzero, s.priv_column().size());
    EXPECT_EQ(rows+1, s.priv_row_start().size());
    }
    for (double x = 0.0; x <= 1.0; x+= 0.1) {
      Sparse<elt_t> s = Sparse<elt_t>::random(rows, cols, x);
      EXPECT_EQ(rows, s.rows());
      EXPECT_EQ(cols, s.columns());
      Tensor<elt_t> t = full(s);
      EXPECT_TRUE(all_equal(Sparse<elt_t>(t), s));
      tensor::index zero = std::count(t.begin(), t.end(), number_zero<elt_t>());
      tensor::index nonzero = t.size() - zero;
      EXPECT_EQ(nonzero, s.length());
      EXPECT_EQ(nonzero, s.priv_data().size());
      EXPECT_EQ(nonzero, s.priv_column().size());
      EXPECT_EQ(rows+1, s.priv_row_start().size());
    }
  }

  TEST(RSparseTest, RSparseRandom) {
    test_over_fixed_rank_tensors<double>(test_sparse_random<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseRandom) {
    test_over_fixed_rank_tensors<cdouble>(test_sparse_random<cdouble>, 2, 7);
  }

  //
  // SPARSE <-> FULL CONVERSION, ARBITRARY SIZES
  //
  template<typename elt_t>
  void test_full(Tensor<elt_t> &t) {
    for (int times = 0; times <= std::min<int>(100,t.size()); times++) {
      tensor::index zeros = std::count(t.begin(), t.end(), number_zero<elt_t>());
      tensor::index nonzeros = t.size() - zeros;
      Tensor<elt_t> tcopy = t;
      Sparse<elt_t> s(t);

      unchanged(tcopy, t);
      EXPECT_EQ(nonzeros, s.length());
      EXPECT_TRUE(all_equal(t, full(s)));
      EXPECT_EQ(nonzeros, s.priv_column().size());
      EXPECT_EQ(nonzeros, s.priv_data().size());
      EXPECT_EQ(t.rows()+1, s.priv_row_start().size());

      if (t.size()) {
        tensor::index i = rand<int>(0, t.rows()-1);
        tensor::index j = rand<int>(0, t.columns()-1);
        t.at(i,j) = number_zero<elt_t>();
      } else {
        break;
      }
    }
  }

  TEST(RSparseTest, RSparseFull) {
    test_over_fixed_rank_tensors<double>(test_full<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseFull) {
    test_over_fixed_rank_tensors<cdouble>(test_full<cdouble>, 2, 7);
  }

  //
  // SPARSE -> COMPLEX CONVERSION, ARBITRARY SIZES
  //
  template<typename elt_t>
  void test_to_complex(Tensor<elt_t> &t) {
    Sparse<elt_t> s = Sparse<elt_t>::random(t.rows(), t.columns());
    Sparse<cdouble> sc = to_complex(s);
    EXPECT_EQ(t.rows(), sc.rows());
    EXPECT_EQ(t.columns(), sc.columns());
    EXPECT_TRUE(all_equal(to_complex(full(s)), full(sc)));
    EXPECT_TRUE(all_equal(s.priv_row_start(), sc.priv_row_start()));
    EXPECT_TRUE(all_equal(s.priv_column(), sc.priv_column()));
  }

  TEST(RSparseTest, RSparseToComplex) {
    test_over_fixed_rank_tensors<double>(test_to_complex<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseToComplex) {
    test_over_fixed_rank_tensors<cdouble>(test_to_complex<cdouble>, 2, 7);
  }


  //
  // REAL PART OF SPARSE MATRICES
  //

  template<typename elt_t>
  void test_real(Tensor<elt_t> &t) {
    Sparse<elt_t> s = Sparse<elt_t>::random(t.rows(), t.columns());
    Sparse<double> r = real(s);
    EXPECT_EQ(t.rows(), r.rows());
    EXPECT_EQ(t.columns(), r.columns());
    EXPECT_TRUE(all_equal(real(full(s)), full(r)));
  }

  TEST(RSparseTest, RSparseReal) {
    test_over_fixed_rank_tensors<double>(test_real<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseReal) {
    test_over_fixed_rank_tensors<cdouble>(test_real<cdouble>, 2, 7);
  }

  //
  // IMAGINARY PART OF SPARSE MATRICES
  //

  template<typename elt_t>
  void test_imag(Tensor<elt_t> &t) {
    Sparse<elt_t> s = Sparse<elt_t>::random(t.rows(), t.columns());
    Sparse<double> i = imag(s);
    EXPECT_EQ(t.rows(), i.rows());
    EXPECT_EQ(t.columns(), i.columns());
    EXPECT_TRUE(all_equal(imag(full(s)), full(i)));
  }

  TEST(RSparseTest, RSparseImag) {
    test_over_fixed_rank_tensors<double>(test_imag<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseImag) {
    test_over_fixed_rank_tensors<cdouble>(test_imag<cdouble>, 2, 7);
  }

  //
  // IMAGINARY PART OF SPARSE MATRICES
  //

  template<typename elt_t>
  void test_conj(Tensor<elt_t> &t) {
    Sparse<elt_t> s = Sparse<elt_t>::random(t.rows(), t.columns());
    Sparse<elt_t> c = conj(s);
    EXPECT_EQ(t.rows(), c.rows());
    EXPECT_EQ(t.columns(), c.columns());
    EXPECT_TRUE(all_equal(conj(full(s)), full(c)));
  }

  TEST(RSparseTest, RSparseConj) {
    test_over_fixed_rank_tensors<double>(test_conj<double>, 2, 7);
  }

  TEST(CSparseTest, CSparseConj) {
    test_over_fixed_rank_tensors<cdouble>(test_conj<cdouble>, 2, 7);
  }

} // namespace tensor_test
