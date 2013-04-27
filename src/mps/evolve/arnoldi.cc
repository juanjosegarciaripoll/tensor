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

#include <tensor/linalg.h>
#include <mps/mps_algorithms.h>
#include <mps/time_evolve.h>

namespace mps {

  using namespace linalg;

  ArnoldiSolver::ArnoldiSolver(const Hamiltonian &H, cdouble dt, int nvectors) :
    TimeSolver(dt), H_(H, 0.0), max_states_(nvectors)
  {
    if (max_states_ <= 0 || max_states_ >= 30) {
      std::cerr << "In ArnoldiSolver(...), the number of states exceeds the limits [1,30]"
		<< std::endl;
      abort();
    }
  }

  double
  ArnoldiSolver::one_step(CMPS *psi, index Dmax)
  {
    CTensor N = CTensor::zeros(max_states_, max_states_);
    CTensor Heff = N;

    std::vector<CMPS> states;
    states.reserve(max_states_);
    states.push_back(*psi);
    N.at(0,0) = to_complex(1.0);
    Heff.at(0,0) = expected(*psi, H_);

    std::vector<CMPS> vectors(3);
    std::vector<cdouble> coeffs(3);
    int nstates;
    for (nstates = 1; nstates < max_states_; nstates++) {
      const CMPS &last = states[nstates-1];
      //
      // 0) Estimate a new vector of the basis.
      //	current = H v[0] - <v[1]|H|v[1]> v[1] - <v[2]|H|v[2]> v[2]
      //    where
      //	v[0] = states[nstates-1]
      //	v[1] = states[nstates-2]
      //
      states.push_back(apply(H_, last));
      CMPS &current = states[nstates-1];
      {
	vectors.clear();
	coeffs.clear();

	vectors.push_back(current);
	coeffs.push_back(number_one<cdouble>());

	vectors.push_back(last);
	coeffs.push_back(- Heff(nstates-1, nstates-1));

	if (nstates > 1) {
	  vectors.push_back(states[nstates-2]);
	  coeffs.push_back(- Heff(nstates-2, nstates-1));
	}
	truncate(&current, vectors[0], 2*Dmax, false);
	simplify(&current, vectors, coeffs, NULL, 2, true);
      }

      //
      // 1) Add the matrices of the new vector to the whole set and, at the same time
      //    compute the scalar products of the old vectors with the new one.
      //    Also compute the matrix elements of the Hamiltonian in this new basis.
      //
      for (int n = 0; n < nstates; n++) {
	cdouble aux;
	N.at(n, nstates) = aux = scprod(states[n], current);
	N.at(nstates, n) = conj(aux);
	Heff.at(n, nstates) = aux = expected(states[n], H_, current);
	Heff.at(nstates, n) = conj(aux);
      }
      N.at(nstates, nstates) = number_one<cdouble>();
      Heff.at(nstates, nstates) = expected(current, H_);
    }
    /**///std::cout << "N=\n"; show_matrix(std::cout, N);
    /**///std::cout << "H=\n"; show_matrix(std::cout, Heff);

    //
    // 2) Once we have the basis, we compute the exponential on it. Notice that, since
    //    our set of states is not orthonormal, we have to first orthogonalize it, then
    //    compute the exponential and finally move on to the original basis and build
    //    the approximate vector.
    //
    CTensor coef = CTensor::zeros(nstates);
    cdouble idt = to_complex(0, -1) * time_step();
    coef.at(0) = to_complex(1.0);
    coef = mmult(expm(idt * solve_with_svd(N, Heff)), coef);

    //
    // 4) Here is where we perform the truncation from our basis to a single MPS.
    //
    return simplify(psi, states, coef, NULL, Dmax, 12);
  }

} // namespace mps
