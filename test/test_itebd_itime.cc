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
#include <tensor/io.h>
#include <mps/quantum.h>
#include <mps/itebd.h>

namespace tensor_test {

  using namespace mps;
  using namespace tensor;

  template<class Tensor> const Tensor flip(const Tensor &x) {
    return mmult(mps::Pauli_x, x);
  }

  template<class Tensor>
  void test_itime_ising_roman()
  {
    Tensor H12 = kron(Pauli_z, Pauli_z);
    Tensor Hmag = (kron(Pauli_x, Pauli_id) + kron(Pauli_id, Pauli_x)) / 2.0;

    struct {
      double hfield;
      RTensor spectrum;
    } tests[] = {
      {
        2.0,
        rgen
        << 0.982473151085781 << 0.017521090616271 << 0.000005655597930
        << 0.000000100860002 << 0.000000001807198 << 0.000000000032229
        << 0.000000000000567 << 0.000000000000010 << 0.000000000000010
        << 0.000000000000000
      },
      {
        1.5,
        rgen
        << 0.964780517745587 <<  0.035170732285978 <<  0.000046973453949
        <<  0.000001712400637 <<  0.000000061771841 <<  0.000000002251871
        <<  0.000000000083843 <<  0.000000000003056 <<  0.000000000003008
        <<  0.000000000000112 <<  0.000000000000110 <<  0.000000000000004
        <<  0.000000000000004 <<  0.000000000000000
      },
      {
        1.3,
        rgen
        << 0.947315010161334 << 0.052514406137734 << 0.000161126882855
        << 0.000008932068504 << 0.000000495539401 << 0.000000027470226
        << 0.000000001559185 << 0.000000000086433 << 0.000000000084285
        << 0.000000000004806 << 0.000000000004672 << 0.000000000000266
        << 0.000000000000265 << 0.000000000000015 << 0.000000000000015
        << 0.000000000000001 << 0.000000000000001 << 0.000000000000001
        << 0.000000000000000
      },
      {
        1.2,
        rgen
        << 0.931507600223784 << 0.068101050999743 << 0.000362709224845
        << 0.000026517099177 << 0.000001966382189 << 0.000000143759099
        << 0.000000010644445 << 0.000000000778199 << 0.000000000765667
        << 0.000000000057816 << 0.000000000055977 << 0.000000000004227
        << 0.000000000004145 << 0.000000000000310 << 0.000000000000303
        << 0.000000000000023 << 0.000000000000023 << 0.000000000000022
        << 0.000000000000002 << 0.000000000000002 << 0.000000000000002
        << 0.000000000000000
      },
      {
        1.1,
        rgen
        << 0.902297502214946 << 0.096466955678060 << 0.001103083986172
        << 0.000117933557105 << 0.000012955878448 << 0.000001385146417
        << 0.000000148073816 << 0.000000015838925 << 0.000000015830954
        << 0.000000001694582 << 0.000000001693380 << 0.000000000181172
        << 0.000000000181024 << 0.000000000019354 << 0.000000000018715
        << 0.000000000002126 << 0.000000000002072 << 0.000000000002001
        << 0.000000000000227 << 0.000000000000221 << 0.000000000000203
        << 0.000000000000024 << 0.000000000000023 << 0.000000000000022
        << 0.000000000000003 << 0.000000000000003 << 0.000000000000002
        << 0.000000000000002 << 0.000000000000000
      }};

    Tensor A = Tensor::random(2);
    iTEBD<Tensor> psi(A, A);
    for (int j = 0; j < 5; j++) {
      std::cout << "============================================================\n"
                << "Bfield = " << tests[j].hfield << std::endl;
      RTensor spectrum = tests[j].spectrum;
      tensor::index max_chi = spectrum.size() + 8;
      tensor::index nsteps = 1000;
      double tolerance = -1;
      Tensor H = -H12 + tests[j].hfield * Hmag;
      double dt;
      int delta;
      for (delta = 20, dt = 0.1; dt > 0.001; dt /= 2, delta *= 2) {
        psi = evolve_itime(psi, H, dt, nsteps, tolerance, max_chi, delta);
	nsteps *= 2;
      }
      Tensor schmidt = psi.schmidt(0);
      tensor::index d = std::min(schmidt.size(), spectrum.size());
      Tensor v1 = schmidt(range(0,d-1));
      Tensor v2 = spectrum(range(0,d-1));
      std::cout << "v1=" << v1 << std::endl << "v2=" << v2 << std::endl;
      std::cout << "n=" << norm2(v1 - v2) << std::endl;
      EXPECT_TRUE(norm2(v1 - v2) < 5e-4);
    }
  }

  template<class Tensor>
  void test_itime_ising()
  {
    tensor::index max_chi = 30;
    double tolerance = -1;

    std::cout << "======================================================================\n"
	      << "UNIFORM MAGNETIC FIELD\n\n";
    {
      Tensor A = Tensor::random(2);
      iTEBD<Tensor> psi(A, A);
      Tensor H12 = kron(Pauli_z, Pauli_id) + kron(Pauli_id, Pauli_z);
      evolve_itime(psi, H12, 0.1, 100, tolerance, max_chi);
    }

    std::cout << "======================================================================\n"
	      << "ANTIFERROMAGNETIC ISING\n\n";
    {
      Tensor A = Tensor::random(2);
      iTEBD<Tensor> psi(A, flip(A));
      Tensor H12 = -kron(Pauli_z, Pauli_z);
      evolve_itime(psi, H12, 0.1, 100, tolerance, max_chi);
    }

    std::cout << "======================================================================\n"
	      << "FERROMAGNETIC ISING\n\n";
    {
      Tensor A = Tensor::random(2);
      iTEBD<Tensor> psi(A, A);
      Tensor H12 = -4.0 * kron(Pauli_z, Pauli_z);
      evolve_itime(psi, H12, 0.2, 100, tolerance, max_chi);
    }

    std::cout << "======================================================================\n"
	      << "FERROMAGNETIC HEISENBERG\n\n";
    {
      Tensor A = Tensor::random(2);
      iTEBD<Tensor> psi(A, A);
      Tensor H12 = -kron(Pauli_x, Pauli_x) - real(kron(Pauli_y, Pauli_y))
	- kron(Pauli_z, Pauli_z);
      evolve_itime(psi, H12, 0.2, 100, tolerance, max_chi);
    }

    std::cout << "======================================================================\n"
	      << "FERROMAGNETIC ISING + TRANSVERSE FIELD\n\n";
    RTensor h(linspace(0.0001, 1.2, 25));
    Tensor S = h;
    Tensor Sx = kron(Pauli_x,Pauli_id) + kron(Pauli_id,Pauli_x);
    Tensor Sz = kron(Pauli_z,Pauli_id) + kron(Pauli_id,Pauli_z);
    Tensor A = Tensor::random(2);
    iTEBD<Tensor> psi(A, A);
    for (size_t i = 0; i < h.size(); i++) {
      Tensor H12 = -4.0 * kron(Pauli_z, Pauli_z) + h[i] * Sx;
      std::cout << "......................................................................\n";
      size_t nsteps = 1000;
      for (double dt = 0.1; dt > 0.01; dt /= 2) {
	psi = evolve_itime(psi, H12, dt, nsteps, tolerance, max_chi, 20);
	std::cout << "...\n";
	nsteps *= 2;
      }
      S.at(i) = real(expected(psi, Pauli_z, Pauli_z));
      std::cout << "Sz=" << S.at(i) << std::endl;
    }
    std::cout << "h=" << h;
    std::cout << "S=" << h;
  }


  ////////////////////////////////////////////////////////////
  /// ITEBD WITH REAL TENSORS
  ///

  TEST(RiTEBDTest, RiTEBDItimeIsing) {
    test_itime_ising<RTensor>();
  }

  TEST(RiTEBDTest, RiTEBDItimeIsingRoman) {
    test_itime_ising_roman<RTensor>();
  }

}
