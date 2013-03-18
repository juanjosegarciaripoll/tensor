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

#include <mps/quantum.h>

namespace mps {

  template <class sparse, class tensor>
  static sparse
  do_sparse_1d_hamiltonian(const std::vector<sparse> &H12,
			   const std::vector<sparse> &H1, bool periodic)
  {
    index N = std::max(H12.size(), H1.size());
    Indices d(N), Dleft(N), Dright(N);
    //
    // We guess the dimension of the physical system from the size of the interaction
    // matrices H12 and the size of H1.
    //
    index D = 1;
    for (index k = 0; k < N; k++) {
      if (!H1[k].is_empty()) {
	d.at(k) = H1[k].rows();
      } else if (!H12[k].is_empty()) {
	d.at(k) = (int)sqrt((double)H12[k].rows());
      } else {
	std::cerr << "In pair_Hamiltonian(H12[], H1[], N), you have not supplied any\n"
		  << "matrix for site " << k;
	abort();
      }
      D *= d[k];
    }
    sparse output(D, D, 0);
    //
    // Dleft[k] is the product of the dimensions of all matrices to the left of
    // the site k
    //
    Dleft.at(0) = 1;
    for (index k = 1; k < N; k++) {
      Dleft.at(k) = Dleft[k-1] * d[k-1];
    }
    //
    // Dright[k] is the product of the dimensions of all matrices to the right of
    // the site k
    //
    Dright.at(N-1) = 1;
    for (index k = N-1; k > 0; k--) {
      Dright.at(k-1) = Dright[k]*d[k];
    }

    for (index k = 1; k < N; k++) {
      if (!H12[k-1].is_empty()) {
	output = output + kron(sparse::eye(Dright[k]),
			       kron(H12[k-1], sparse::eye(Dleft[k-1])));
      }
    }
    if (periodic && (N > 1)) {
      tensor O1, O2;
      decompose_operator(full(H12[N-1]), &O1, &O2);
      sparse aux = sparse::eye((index)Dleft[N-1]/d[0]);
      for (index i = 0; i < O1.dimension(-1); i++) {
	sparse sO1 = sparse(squeeze(O1(range(),range(),range(i))));
	sparse sO2 = sparse(squeeze(O2(range(),range(),range(i))));
	output = output + kron(sO2, kron(aux, sO1));
      }
    }
    for (index k = 0; k < N; k++) {
      if (!H1[k].is_empty()) {
	output = output + kron(sparse::eye(Dright[k]),
			       kron(H1[k], sparse::eye(Dleft[k])));
      }
    }
    return output;
  }

} // namespace mps
