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
   * Trotter's method with three passes
   *
   *	exp(-iHdt) = exp(-iH_even dt/2) exp(-iH_odd dt) exp(-iH_even dt/2)
   */

  Trotter3Solver::Trotter3Solver(const Hamiltonian &H, cdouble dt,
                                 bool do_optimize, double tol) :
  TrotterSolver(dt), U1(H, 1, dt, true), U2(H, 0, dt/2.0, true),
    optimize(do_optimize),
    sweeps(32), normalize(true), sense(0), tolerance(tol)
  {
  }

  double
  Trotter3Solver::one_step(CMPS *P, index Dmax)
  {
    if (sense == 0) {
      if (normalize) {
        *P = normal_form(*P);
      } else {
        *P = canonical_form(*P);
      }
      sense = +1;
    }
    double err;
    if (optimize) {
      CMPS Pfull = *P;
      err = U2.apply(&Pfull, sense, Dmax, true); sense = -sense;
      err += U1.apply(&Pfull, sense, Dmax, true); sense = -sense;
      err += U2.apply(&Pfull, sense, Dmax, true); sense = -sense;
      if (truncate(P, Pfull, Dmax, false)) {
        return simplify(P, Pfull, &sense, false, sweeps, normalize);
      }
    } else {
      err = U2.apply(P, sense, Dmax, true); sense = -sense;
      err += U1.apply(P, sense, Dmax, true); sense = -sense;
      err += U2.apply(P, sense, Dmax, true); sense = -sense;
    }
    return err;
  }

} // namespace mps
