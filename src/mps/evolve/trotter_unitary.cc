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
#include <mps/time_evolve.h>
#include <tensor/io.h>

namespace mps {

  TrotterSolver::Unitary::Unitary(const Hamiltonian &H, index k, cdouble dt,
                                  bool apply_pairwise, bool do_debug,
                                  double the_tolerance) :
    debug(do_debug), pairwise(apply_pairwise), tolerance(the_tolerance),
    k0(k), kN(H.size()), U(H.size())
  {
    if (the_tolerance >= 0.0)
      abort();
    if (k > 1) {
      std::cerr << "In TrotterSolver::Unitary::Unitary(H, k, ...), "
        "the initial site k was neither 0 nor 1";
      abort();
    }
    if (pairwise) {
      if ((k & 1) ^ (kN & 1)) {
	// We look for the nearest site which has the same parity.
	kN--;
      }
      if ((kN - k0) < 2) {
	// No sites!
	kN = k0;
      }
    } else {
      kN = kN - 1;
    }
    if (debug) std::cout << "computing: ";
    dt = to_complex(-abs(imag(dt)), -real(dt));
    for (int di = 1, i = 0; i < (int)H.size(); i += di) {
      CTensor Hi;
      if (i < k0 || i >= kN) {
	if (!pairwise) continue;
	// Local operator
	Hi = H.local_term(i,0.0) / 2.0;
	if (debug) std::cout << "[" << i << "]";
	di = 1;
      } else {
	double f1 = 0.5;
	double f2 = 0.5;
	if (pairwise) {
	  di = 2;
	} else {
	  if (i == 0) f1 = 1.0;
	  if (i+2 == (int)H.size()) f2 = 1.0;
	  di = 1;
	}
	CTensor i1 = CTensor::eye(H.dimension(i));
	CTensor i2 = CTensor::eye(H.dimension(i+1));
	Hi = H.interaction(i,0.0)
	  + kron2(H.local_term(i,0.0) * f1, i2)
	  + kron2(i1, H.local_term(i+1,0.0) * f2);
	if (debug) std::cout << "[" << i << "," << i+1 << "]";
      }
      U.at(i) = linalg::expm(Hi * dt);
    }
    if (debug) {
      std::cout << std::endl;
      std::cout << "Unitaries running from " << k0 << " to " << kN << std::endl;
    }
  }

  void
  TrotterSolver::Unitary::apply_onto_one_site(CMPS &P, const CTensor &Uloc, index k, int dk, bool guifre) const
  {
    CTensor P1 = P[k];
    if (Uloc.is_empty()) {
      if (debug) {
	std::cout << "<" << k << ">";
      }
    } else {
      if (debug) {
	std::cout << "(" << k << ")";
      }
      index a1,i1,a2;
      P1.get_dimensions(&a1, &i1, &a2);
      P1 = foldin(Uloc, -1, P1, 1);
    }
    set_canonical(P, k, P1, dk);
  }

  double
  TrotterSolver::Unitary::apply_onto_two_sites(CMPS &P, const CTensor &U12,
					       index k1, index k2, int dk,
					       index max_a2, bool guifre) const
  {
    index a1, i1, a2, i2, a3;

    CTensor P1 = P[k1];
    P1.get_dimensions(&a1, &i1, &a2);
    CTensor P2 = P[k2];
    P2.get_dimensions(&a2, &i2, &a3);

    double err = 0.0;
    if (U12.is_empty()) {
      if (debug) {
	std::cout << "<" << k1 << "," << k2 << ">";
      }
    } else {
      if (debug) {
	std::cout << "(" << k1 << "," << k2 << ")";
      }
      /* Apply the unitary onto two neighboring sites. This creates a
       * larger tensor that we have to split into two new tensors, Pout[k1]
       * and Pout[k2], that represent the sites */
      P1 = reshape(fold(P1, -1, P2, 0), a1,i1*i2,a3);
      P1 = foldin(U12, -1, P1, 1);
      RTensor s = linalg::svd(reshape(P1,a1*i1,i2*a3), &P1, &P2, SVD_ECONOMIC);
      if (dk > 0) {
	scale_inplace(P2, 0, s);
      } else {
	scale_inplace(P1, -1, s);
      }
      a2 = s.size();
      /*
       * Here we perform a first truncation of the matrix.
       *   1) If we use Guifre's algorithm, the truncation is based on the
       *   maximum size that we want to keep, or a tolerance.
       *   2) If we on the other hand use the simplify() routine, we just
       *   remove the zero elements which appear from the zeros in the
       *   singular value decomposition above.
       * Notice that at the end, P is orthonormalized in both cases.
       */
      index new_a2 = where_to_truncate(s, tolerance, max_a2? max_a2 : a2);
      if (debug) {
        std::cout << "a2=" << a2 << ", new_a2=" << new_a2
                  << ", tol=" << tolerance
                  << ", max=" << (max_a2? max_a2 : a2)
                  << ", s=" << s << std::endl;
      }
      if (new_a2 != a2) {
	P1 = change_dimension(P1, -1, new_a2);
	P2 = change_dimension(P2, 0, new_a2);
	a2 = new_a2;
        for (index i = a2; i < s.size(); i++)
          err += square(s[i]);
      }
    }
    if (dk > 0) {
      P.at(k1) = reshape(P1, a1,i1,a2);
      set_canonical(P, k2, reshape(P2, a2,i2,a3), dk);
    } else {
      P.at(k2) = reshape(P2, a2,i2,a3);
      set_canonical(P, k1, reshape(P1, a1,i1,a2), dk);
    }
    return err;
  }

  double
  TrotterSolver::Unitary::apply(CMPS *psi, int sense, index Dmax, bool guifre, bool normalize) const
  {
    index L = psi->size();
    double err = 0;
    int dk = pairwise? 2 : 1;
    if (sense > 0) {
      if (pairwise) {
	for (int k = 0; k < k0; k++) {
	  apply_onto_one_site(*psi, U[k], k, sense, guifre);
	}
      }
      for (int k = k0; k < kN; k += dk) {
	err += apply_onto_two_sites(*psi, U[k], k, k+1, sense, Dmax, guifre);
      }
      if (pairwise) {
	for (int k = kN; k < (int)L; k++) {
	  apply_onto_one_site(*psi, U[k], k, sense, guifre);
	}
      }
    } else {
      if (pairwise) {
	for (int k = L-1; k >= kN; k--) {
	  apply_onto_one_site(*psi, U[k], k, sense, guifre);
	}
      }
      for (int k = kN - dk; k >= k0; k -= dk) {
	err += apply_onto_two_sites(*psi, U[k], k, k+1, sense, Dmax, guifre);
      }
      if (pairwise) {
	for (int k = k0 - 1; k >= 0; k--) {
	  apply_onto_one_site(*psi, U[k], k, sense, guifre);
	}
      }
    }
    if (debug) {
      std::cout << std::endl;
    }
    if (normalize) {
      index k = (sense > 0)? L-1 : 0;
      CTensor Pk = (*psi)[k];
      psi->at(k) = Pk / norm2(Pk);
    }
    return err;
  }

} // namespace mps

