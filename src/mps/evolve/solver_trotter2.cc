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

#include <mps/time_evolve.h>
#include <mps/mps_algorithms.h>

namespace mps {

  /**********************************************************************
   * Trotter's method with only two passes.
   *
   *	exp(-iHdt) = \prod_k={N-1}^1 exp(-iH_{kk+1} dt/2) \prod_k=1^{N-1} exp(-iH_{kk+1} dt/2)
   */

  Trotter2Solver::Trotter2Solver(const Hamiltonian &H, cdouble dt, bool do_optimize,
				 double tol) :
  TrotterSolver(dt), U(H, 0, dt/2.0, false, false, tol),
    optimize(do_optimize), sense(0),
    sweeps(32), normalize(true), tolerance(tol)
  {
  }

  double
  Trotter2Solver::one_step(CMPS *P, index Dmax)
  {
    if (sense == 0) {
      if (normalize) {
	*P = normal_form(*P);
      } else {
	*P = canonical_form(*P);
      }
      sense = +1;
    }
    if (optimize) {
      CMPS Pfull = *P;
      U.apply(&Pfull, sense, 0, false); sense = -sense;
      U.apply(&Pfull, sense, 0, false, true); sense = -sense;
      if (truncate(P, Pfull, Dmax, false)) {
	return simplify(P, Pfull, &sense, false, sweeps, normalize);
      }
      return 0.0;
    } else {
      double err = U.apply(P, sense, Dmax, true); sense = -sense;
      err += U.apply(P, sense, Dmax, true, true); sense = -sense;
      return err;
    }
  }

} // namespace mps
