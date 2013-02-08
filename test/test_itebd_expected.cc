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
#include <gtest/gtest-death-test.h>
#include <mps/quantum.h>
#include <mps/itebd.h>

namespace tensor_test {

  using namespace mps;
  using namespace tensor;

  template<class Tensor>
  const iTEBD<Tensor> random_product(int d, bool same = false, Tensor *pA=0, Tensor *pB=0)
  {
      Tensor A = Tensor::random(d);
      Tensor B = same? A : Tensor::random(d);
      Tensor nA = A / norm2(A);
      Tensor nB = B / norm2(B);
      if (pA) *pA = nA;
      if (pB) *pB = nB;
      return iTEBD<Tensor>(A, B);
  }

  template<class Tensor>
  void test_expected_product_norm(int d)
  {
    for (int i = 1; i < 100; i++) {
      Tensor id = Tensor::eye(d,d);
      typename Tensor::elt_t one = id(0,0);
      iTEBD<Tensor> psiAA = random_product<Tensor>(d, true);
      iTEBD<Tensor> psiAB = random_product<Tensor>(d, false);
      /* The expected value of the identity is the norm */
      EXPECT_TRUE(simeq(one, expected(psiAA, id)));
      EXPECT_TRUE(simeq(one, expected(psiAB, id)));
      /* Test translational invariance */
      EXPECT_CEQ(expected(psiAA, id, 0), expected(psiAA, id, 1));
      EXPECT_CEQ(expected(psiAA, id, 0), expected(psiAA, id, 2));
      EXPECT_CEQ(expected(psiAA, id, 1), expected(psiAA, id, -1));
      /* Test translational invariance */
      EXPECT_CEQ(expected(psiAB, id, 0), expected(psiAA, id, 2));
      EXPECT_CEQ(expected(psiAB, id, 1), expected(psiAA, id, 3));
      EXPECT_CEQ(expected(psiAB, id, 1), expected(psiAA, id, -1));
    }
  }

  template<class Tensor>
  Tensor projector(const Tensor &A)
  {
    const Tensor Am = reshape(A, A.size(), 1);
    return mmult(Am, adjoint(Am));
  }

  /*
   * When computing correlations, ensure that operators act on the right
   * sites and that expectation values are translationally invariant with
   * period two.
   */
  template<class Tensor>
  void test_expected_projectors(int d)
  {
    for (int i = 1; i < 100; i++) {
      Tensor A, B;
      iTEBD<Tensor> psi = random_product<Tensor>(d, true, &A, &B);
      Tensor PA = projector(A); // Projector onto A
      Tensor PB = projector(B); // Projector onto B
      Tensor id = Tensor::eye(d,d);
      Tensor PnA = id - PA; // Orthogonal projectors
      Tensor PnB = id - PB;
      typename Tensor::elt_t one = number_one<typename Tensor::elt_t>();
      typename Tensor::elt_t zero = number_zero<typename Tensor::elt_t>();

      EXPECT_CEQ(one, expected(psi, PA, 0));
      EXPECT_CEQ(zero, expected(psi, PnA, 0));
      EXPECT_CEQ(one, expected(psi, PB, 1));
      EXPECT_CEQ(zero, expected(psi, PnB, 1));

      EXPECT_CEQ(one, expected(psi, PA, PB));
      EXPECT_CEQ(zero, expected(psi, PA, PnB));
      EXPECT_CEQ(zero, expected(psi, PnA, PB));
      EXPECT_CEQ(zero, expected(psi, PnA, PnB));

      EXPECT_CEQ(one, expected(psi, PA, id));
      EXPECT_CEQ(one, expected(psi, id, PB));
      EXPECT_CEQ(zero, expected(psi, id, PnB));
      EXPECT_CEQ(zero, expected(psi, PnA, id));

      EXPECT_CEQ(one, expected(psi, PA, 0));
      EXPECT_CEQ(zero, expected(psi, PnA, 0));
      EXPECT_CEQ(one, expected(psi, PB, 1));
      EXPECT_CEQ(zero, expected(psi, PnB, 1));
    }
  }

  /*
   * Verify the implementation of expected12() by checking with kronecker
   * products of projectors.
   */
  template<class Tensor>
  void test_expected12_projectors(int d)
  {
    for (int i = 1; i < 100; i++) {
      Tensor A, B;
      iTEBD<Tensor> psi = random_product<Tensor>(d, false, &A, &B);
      Tensor PA = projector(A); // Projector onto A
      Tensor PB = projector(B); // Projector onto B
      Tensor id = Tensor::eye(d,d);
      Tensor PnA = id - PA; // Orthogonal projectors
      Tensor PnB = id - PB;
      typename Tensor::elt_t one = number_one<typename Tensor::elt_t>();
      typename Tensor::elt_t zero = number_zero<typename Tensor::elt_t>();

      EXPECT_CEQ(one, expected12(psi, kron2(PA, PB)));
      EXPECT_CEQ(zero, expected12(psi, kron2(PA, PnB)));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnA, PB)));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnA, PnB)));

      EXPECT_CEQ(one, expected12(psi, kron2(PA, id)));
      EXPECT_CEQ(one, expected12(psi, kron2(id, PB)));
      EXPECT_CEQ(zero, expected12(psi, kron2(id, PnB)));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnA, id)));

      EXPECT_CEQ(one, expected12(psi, kron2(PB, PA), 1));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnB, PA), 1));
      EXPECT_CEQ(zero, expected12(psi, kron2(PB, PnA), 1));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnB, PnA), 1));

      EXPECT_CEQ(one, expected12(psi, kron2(id, PA), 1));
      EXPECT_CEQ(one, expected12(psi, kron2(PB, id), 1));
      EXPECT_CEQ(zero, expected12(psi, kron2(PnB, id), 1));
      EXPECT_CEQ(zero, expected12(psi, kron2(id, PnA), 1));
    }
  }

  /*
   * Verify the implementation of expected12() by checking with kronecker
   * products of projectors.
   */
  template<class Tensor>
  void test_energy_projectors(int d)
  {
    for (int i = 1; i < 100; i++) {
      Tensor A, B;
      iTEBD<Tensor> psi = random_product<Tensor>(d, true, &A, &B);
      Tensor PA = projector(A); // Projector onto A
      Tensor id = Tensor::eye(d,d);
      Tensor PnA = id - PA; // Orthogonal projectors
      typename Tensor::elt_t one = number_one<typename Tensor::elt_t>();
      typename Tensor::elt_t zero = number_zero<typename Tensor::elt_t>();

      EXPECT_CEQ(2.0, energy(psi, kron(PA, PA)));
      EXPECT_CEQ(0.0, energy(psi, kron(PA, PnA)));
      EXPECT_CEQ(0.0, energy(psi, kron(PnA, PA)));
      EXPECT_CEQ(0.0, energy(psi, kron(PnA, PnA)));

      EXPECT_CEQ(2.0, energy(psi, kron(PA, id)));
      EXPECT_CEQ(2.0, energy(psi, kron(id, PA)));
      EXPECT_CEQ(0.0, energy(psi, kron(id, PnA)));
      EXPECT_CEQ(0.0, energy(psi, kron(PnA, id)));
    }
  }

  /*
   * Verify the string order parameter computation using the AKLT
   * state, for which it is an exact order.
   */
  template<class Tensor>
  void test_aklt_string_order()
  {
    iTEBD<Tensor> psi = infinite_aklt_state();
    Tensor Sz = RTensor(igen << 3 << 3,
                        rgen << 1 << 0 << 0
                        << 0 << 0 << 0 << 0 << 0 << -1);
    Tensor ExpSz = RTensor(igen << 3 << 3,
                           rgen << -1 << 0 << 0
                           << 0 << 1 << 0 << 0 << 0 << -1);
    double v = 1/sqrt(2.0);
    Tensor Sx = RTensor(igen << 3 << 3,
                        rgen << 0 << v << 0
                        << v << 0 << v << 0 << v << 0);
    Tensor ExpSx = RTensor(igen << 3 << 3,
                           rgen << 0 << 0 << -1
                           << 0 << -1 << 0 << -1 << 0 << 0);
    for (int i = 1; i < 20; i++) {
      typename Tensor::elt_t vz =
        string_order(psi, Sz, 0, ExpSz, Sz, i);
      EXPECT_CEQ(vz, -4.0/9.0);
      typename Tensor::elt_t vx =
        string_order(psi, Sx, 0, ExpSx, Sx, i);
      EXPECT_CEQ(vz, vx);
    }
  }

  ////////////////////////////////////////////////////////////
  /// ITEBD WITH REAL TENSORS
  ///

  TEST(RiTEBDTest, NormProduct) {
    test_over_integers(1, 6, test_expected_product_norm<RTensor>);
  }

  TEST(RiTEBDTest, ExpectedProjectors) {
    test_over_integers(1, 6, test_expected_projectors<RTensor>);
  }

  TEST(RiTEBDTest, Expected12Projectors) {
    test_over_integers(1, 6, test_expected12_projectors<RTensor>);
  }

  TEST(RiTEBDTest, EnergyProjectors) {
    test_over_integers(1, 6, test_energy_projectors<RTensor>);
  }

  TEST(RiTEBDTest, AKLTStringOrder) {
    test_aklt_string_order<RTensor>();
  }

  ////////////////////////////////////////////////////////////
  /// ITEBD WITH COMPLEX TENSORS
  ///

  TEST(CiTEBDTest, NormProduct) {
    test_over_integers(1, 6, test_expected_product_norm<CTensor>);
  }

  TEST(CiTEBDTest, ExpectedProjectors) {
    test_over_integers(1, 6, test_expected_projectors<CTensor>);
  }

  TEST(CiTEBDTest, Expected12Projectors) {
    test_over_integers(1, 6, test_expected12_projectors<CTensor>);
  }

  TEST(CiTEBDTest, EnergyProjectors) {
    test_over_integers(1, 6, test_energy_projectors<CTensor>);
  }

  TEST(CiTEBDTest, AKLTStringOrder) {
    test_aklt_string_order<CTensor>();
  }

}
