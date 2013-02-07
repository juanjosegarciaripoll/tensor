// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <numeric>
#include <tensor/linalg.h>
#include <tensor/io.h>
#include <mps/itebd.h>
#include "itebd_expected_slow.hpp"

namespace mps {

  static inline bool
  stop(double delta, double tol, double value)
  {
    return delta < tol * std::max(abs(value), 1e-3);
  }

  static inline double
  avg_change(size_t step, RTensor &cum, double delta)
  {
    size_t L = cum.size();
    size_t i = step % L;
    cum.at(i) = delta;
    if (step >= L) {
      return std::accumulate(cum.begin(), cum.end(), 0.0);
    } else {
      return 1.0;
    }
  }

  template<class Tensor>
  const iTEBD<Tensor>
  evolve_itime(iTEBD<Tensor> psi, const Tensor &H12,
	       double dt, tensor::index nsteps,
	       double tolerance, tensor::index max_dim,
	       tensor::index deltan)
  {
    static const double FR_param[5] =
      {0.67560359597983, 1.35120719195966, -0.17560359597983, -1.70241438391932};

    Tensor eH12[4];
    int method = 2;
    switch (method) {
    case 1:
      /* Second order Trotter expansion */
      eH12[1] = linalg::expm((-dt/2) * H12);
    case 0:
      /* First order Trotter expansion */
      eH12[0] = linalg::expm((-dt) * H12);
      break;
    default:
      /* Fourth order Trotter expansion */
      for (int i = 0; i < 4; i++) {
	eH12[i] = linalg::expm((-dt*FR_param[i]) * H12);
      }
    }
    Tensor Id = Tensor::eye(H12.rows());
    double time = 0;
    double E = energy(psi, H12), S = psi.entropy();
    std::cout.precision(5);
    std::cout << nsteps << ", " << dt << " x " << deltan << " = " << dt * deltan << std::endl;
    RTensor S_growth((deltan < 10)? 10 : deltan);
    RTensor E_growth((deltan < 10)? 10 : deltan);
    S_growth.fill_with_zeros();
    E_growth.fill_with_zeros();
    bool stop = false;
    for (size_t i = 0; (i < nsteps) && (!stop); i++) {
      switch (method) {
      case 0:
	psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
	psi = psi.apply_operator(eH12[0], 1, tolerance, max_dim);
	break;
      case 1:
	psi = psi.apply_operator(eH12[1], 0, tolerance, max_dim);
	psi = psi.apply_operator(eH12[0], 1, tolerance, max_dim);
	psi = psi.apply_operator(eH12[1], 0, tolerance, max_dim);
	break;
      default:
	psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
	psi = psi.apply_operator(eH12[1], 1, tolerance, max_dim);
	psi = psi.apply_operator(eH12[2], 0, tolerance, max_dim);
	psi = psi.apply_operator(eH12[3], 1, tolerance, max_dim);
	psi = psi.apply_operator(eH12[2], 0, tolerance, max_dim);
	psi = psi.apply_operator(eH12[1], 1, tolerance, max_dim);
	psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
      }
      double newE = energy(psi, H12);
      double newS = psi.entropy();
      double dS = S - newS;
      double dE = E - newE;
      double dSdt = avg_change(i, S_growth, dS) / dt;
      double dEdt = avg_change(i, E_growth, dE) / dt;
      S = newS;
      E = newE;
      time += dt;
      if (i > E_growth.size() && ((abs(dSdt) < 1e-6) && (dEdt <= 1e-6))) {
	std::cout << "Entropy and energy converged" << std::endl;
	stop = true;
      }
      if ((deltan && (i % deltan  == 0)) || stop) {
	std::cout << "t=" << time << ";\tE=" << E << ";\tS=" << S
		  << ";\tl=" << std::max(psi.left_dimension(0),
					 psi.right_dimension(0))
		  << std::endl;
	std::cout << "\tdS=" << dS << ";\tdE=" << dE << std::endl;
	std::cout << "\tdSdt=" << dSdt << ";\tdEdt=" << dEdt << std::endl;
      }
    }
    return psi;
  }

}
