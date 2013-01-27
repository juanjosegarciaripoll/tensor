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

namespace mps {

  //----------------------------------------------------------------------
  // PAIR HAMILTONIAN
  //
  // Given a nearest neighbor interaction term, H12, and a local term, H1,
  // we build a Hamiltonian acting on the lattice, and translationally
  // invariant, by simply displacing H12 and H1.
  //

  template <class sparse, class tensor>
  static const sparse
  do_pair_Hamiltonian(const sparse &H12, const sparse &H1, index N, bool periodic)
  {
    if (H12.is_empty() && H1.is_empty()) {
      std::cerr << "In pair_Hamiltonian(H12, H1, N), all matrices are empty. You must\n"
		<< "supply either an interaction or a local Hamiltonian for each site.";
      abort();
    }
    int d2 = H12.rows();
    int d = std::max((int)H1.rows(), (int)sqrt((double)d2));
    int D = (index)pow((double)d, (double)N);
    sparse output(D, D, 0);

    if (H12.length()) {
      for (index k = 1; k < N; k++) {
	index D1 = (index)pow((double)d, (double)(k-1));
	index D2 = (index)pow((double)d, (double)(N-k-1));
	output = output + kron(sparse::eye(D1), kron(H12, sparse::eye(D2)));
      }
      if (periodic && (N > 1)) {
	tensor O1, O2;
	decompose_operator(full(H12), &O1, &O2);
	sparse aux = sparse::eye((index)pow((double)d, (double)(N-2)));
	for (index i = 0; i < O1.dimension(-1); i++) {
	  sparse sO1 = sparse(squeeze(O1(range(),range(),range(i))));
	  sparse sO2 = sparse(squeeze(O2(range(),range(),range(i))));
	  output = output + kron(sO2, kron(aux, sO1));
	}
      }
    }
    if (H1.length()) {
      for (index k = 0; k < N; k++) {
	index D1 = (index)pow((double)d, (double)k);
	index D2 = (index)pow((double)d, (double)(N-k-1));
	output = output + kron(sparse::eye(D1), kron(H1, sparse::eye(D2)));
      }
    }
    return output;
  }

} // namespace mps
