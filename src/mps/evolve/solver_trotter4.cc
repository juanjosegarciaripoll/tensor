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
   * Trotter's method of fourth order.
   */

  static const double inv_theta = 0.74007895010513;
  static const double FR_param[5] = {0.67560359597983, 1.35120719195966,
                                     -0.17560359597983, -1.70241438391932};

  ForestRuthSolver::ForestRuthSolver(const Hamiltonian &H, cdouble dt, bool do_optimize,
                                     double tol) :
    TrotterSolver(dt),
    U1(H, 0, dt*FR_param[0], true, false, tol),
    U2(H, 1, dt*FR_param[1], true, false, tol),
    U3(H, 0, dt*FR_param[2], true, false, tol),
    U4(H, 1, dt*FR_param[3], true, false, tol),
    optimize(do_optimize),
    sense(0), sweeps(32), normalize(true), debug(0),
    tolerance(tol)
  {
  }

  double
  ForestRuthSolver::one_step(CMPS *P, index Dmax)
  {
    if (sense == 0) {
      if (normalize) {
	*P = normal_form(*P);
      } else {
	*P = canonical_form(*P);
      }
      sense = 1;
    }
    if (optimize) {
      double err = 0;
      CMPS Pfull = *P;
      U1.apply(&Pfull, sense, 0, false); sense = -sense;
      U2.apply(&Pfull, sense, 0, false); sense = -sense;
      if (truncate(P, Pfull, Dmax, false)) {
	err += simplify(P, Pfull, &sense, false, sweeps, normalize);
      }
      Pfull = *P;
      U3.apply(&Pfull, sense, 0, false); sense = -sense;
      U4.apply(&Pfull, sense, 0, false); sense = -sense;
      U3.apply(&Pfull, sense, 0, false); sense = -sense;
      if (truncate(P, Pfull, Dmax, false)) {
	err += simplify(P, Pfull, &sense, false, sweeps, normalize);
      }
      Pfull = *P;
      U2.apply(&Pfull,-1, 0, false); sense = -sense;
      U1.apply(&Pfull, 1, 0, false); sense = -sense;
      if (truncate(P, Pfull, Dmax, false)) {
	err += simplify(P, Pfull, &sense, false, sweeps, normalize);
      }
      return err;
    } else {
      U1.apply(P, sense, Dmax, true); sense = -sense;
      U2.apply(P, sense, Dmax, true); sense = -sense;
      U3.apply(P, sense, Dmax, true); sense = -sense;
      U4.apply(P, sense, Dmax, true); sense = -sense;
      U3.apply(P, sense, Dmax, true); sense = -sense;
      U2.apply(P, sense, Dmax, true); sense = -sense;
      U1.apply(P, sense, Dmax, true); sense = -sense;
      *P = normal_form(*P);
    }

    return 0.0;
  }

} // namespace mps
