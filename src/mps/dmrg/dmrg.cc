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

#include <cmath>
#include <algorithm>
#include <tensor/tools.h>
#include <tensor/linalg.h>
#include <tensor/arpack.h>
#include <tensor/io.h>
#include <mps/tools.h>
#include <mps/mps_algorithms.h>
#include <mps/dmrg.h>

namespace mps {

  #define direct_sum kron2_sum

  /*----------------------------------------------------------------------
   * SEARCH OF GROUND STATES:
   * ========================
   *
   * With respect to a single site, the energy functional of a MPS is the quotient
   * of two quadratic polynomials. If Pk are the matrices of the MPS,
   *	E(Pk) = fH(Pk)/fN(Pk)	fH = <psi|H|psi>, fN = <psi|psi>
   * which means that finding the optimal Pk reduces to solving the eigenvalue
   * problem
   *	fH'' Pk = E fN'' Pk,
   * the colon ('') denoting second order derivative (fH(0)=fN(0)=0).
   *
   * Given a site, k, we can construct the left and right basis and write the
   * state |psi> as
   *	|psi> = P(a,i,b) |c,a>|i>|b,c>
   * where we sum over repeated indices. We only need the following operators:
   *	Hl(c,a;c',a') = Hamiltonian for left block
   *	Hil(c,a,i;c',a',i') = Interaction left-site
   *	Hir(i,b,c;i',b',c') = Interaction site-right
   *	Hr(b,c;b',c') = Hamiltonian for right block
   *	Nl(c,a;c',a') = <c,a|c',a'>
   *	Nr(b,c;b',c') = <b,c|b',c'>
   * A different algorithm uses pairs of sites, as in
   *	|psi> = P(a,i,b)Q(b,j,c) |a>|i>|j>|b>
   */

  template<class MPS>
  DMRG<MPS>::DMRG(const Hamiltonian &H) :
    H_(H.duplicate()), Hl_(H.size()), Hr_(Hl_),
    Ql_(0), Qr_(0),
    full_size_(0),
    valid_cells_(),
    error(false),
    sweeps(32), display(true), debug(0),
    tolerance(1e-6), svd_tolerance(1e-8),
    eigenvalues(), neigenvalues(1),
    Q_values(), Q_operators(0), P0_(), Proj_()
  {
    if (size() < 2) {
      std::cerr << "The DMRG solver only solves problems with more than two sites";
      abort();
    }
    if (H.is_periodic()) {
      std::cerr << "The DMRG class does not work with periodic boundary conditions";
      abort();
    }
  }

  template<class MPS>
  DMRG<MPS>::~DMRG()
  {
    delete H_;
  }

  template<class MPS>
  void
  DMRG<MPS>::clear_orthogonality()
  {
    P0_.clear();
  }

  template<class MPS>
  void
  DMRG<MPS>::orthogonal_to(const MPS &P)
  {
    P0_.push_back(P);
  }

  /**********************************************************************
   * SINGLE-SITE DMRG ALGORITHM
   */

  template<class tensor>
  static inline tensor
  ensure_Hermitian(const tensor &A)
  {
    return (A + adjoint(A))/2.0;
  }

  template<class MPS>
  double
  DMRG<MPS>::minimize_single_site(MPS &P, index k, int dk)
  {
    /* First we build the components of the Hamiltonian:
     *		Op = Opl + Opli + Opi + Opir + Opr
     * which are the Hamiltonian for the left and right block (Opl,Opr) the
     * intearction between block and site (Opli,Opir) and the local terms
     * (Opi). In the case of periodic b.c. we also will need the norm
     * matrices. Notice that, for economy, we join Opl, Opli on one side, and
     * Opr and Opr on the other. Opi is either added to Opli or to Opri
     * depending on which operator we will use in update_matrices_left/right.
     */
    index a1,i1,b1;
    elt_t Pk = P[k];
    Pk.get_dimensions(&a1, &i1, &b1);

    elt_t Opli = block_site_interaction_left(P, k);
    elt_t Opir = block_site_interaction_right(P, k);
    if (dk > 0) {
      Opli = Opli + kron2(elt_t::eye(a1), local_term(k));
    } else {
      Opir = Opir + kron2(local_term(k), elt_t::eye(b1));
    }

    /*
     * Now we find the minimal energy and optimal projector, Pk.  We use an
     * iterative algorithm that does not require us to build explicitely the
     * matrix.  NOTE: For small sizes, our iterative algorithm (ARPACK) fails
     * and we have to resort to a full diagonalization.
     */
    index neig = std::max<int>(1, neigenvalues);
    elt_t aux;
    if (a1*i1*b1 <= 10) {
      elt_t Heff = kron2(Opli, elt_t::eye(b1)) + kron2(elt_t::eye(a1), Opir);
      aux = eigs(Heff, linalg::SmallestAlgebraic, neig, &Pk, Pk.begin());
    } else {
      linalg::Arpack<elt_t> eigs(Pk.size(), linalg::SmallestAlgebraic, neig);
      eigs.set_maxiter(Pk.size());
      eigs.set_start_vector(Pk.begin());
      while (eigs.update() < eigs.Finished) {
	Pk = eigs.get_x();
	Pk = fold(reshape(Opli, a1*i1,a1*i1), -1, reshape(Pk, a1*i1,b1), 0)
	  + fold(reshape(Pk, a1,i1*b1), -1, reshape(Opir, i1*b1,i1*b1), -1);
	eigs.set_y(Pk);
      }
      if (eigs.get_status() != eigs.Finished) {
	std::cerr << "DMRG: Diagonalization routine did not converge.\n"
		  << eigs.error_message();
	abort();
      }
      aux = eigs.get_data(&Pk);
    }
    /*
     * And finally we update the matrices.
     */
    if (neig > 1) Pk = Pk(range(), range(0));
    set_canonical(P, k, reshape(Pk, a1,i1,b1), dk);
    if (dk > 0) {
      update_matrices_left(P, k);
    } else {
      update_matrices_right(P, k);
    }
    eigenvalues = real(aux);
    return eigenvalues[0];
  }

  /**********************************************************************
   * CONSERVED QUANTITIES
   */

  template<class MPS>
  index
  DMRG<MPS>::n_constants() const
  {
    return Q_operators.size();
  }

  template<class MPS>
  void
  DMRG<MPS>::clear_conserved_quantities()
  {
    return Q_operators.clear();
  }

  template<class MPS>
  void
  DMRG<MPS>::commutes_with(const DMRG<MPS>::elt_t &Q)
  {
    Q_operators.push_back(Q);
  }

  template<class MPS>
  void
  DMRG<MPS>::prepare_simplifier(index k, const elt_t &Pij)
  {
    index n_Q = Q_values.size();
    if (n_Q) {
      RTensor Z = RTensor::zeros(1,1);
      Booleans flag;
      for (index n = 0; n < n_Q; n++) {
	RTensor Nl = (k > 0)? real(Ql_[n][k-1]) : Z;
	RTensor Nr = ((k+2)<Qr_[n].size())? real(Qr_[n][k+2]) : Z;
	RTensor Ni = real(take_diag(Q_operators[n]));
	double desired_N = Q_values[n];
	RTensor N = direct_sum(direct_sum(Nl, Ni), direct_sum(Ni, Nr));
	if (n == 0) {
	  flag = (N == desired_N);
	} else {
	  flag = flag && (N == desired_N);
	}
      }
      valid_cells_ = which(flag);
      full_size_ = Pij.size();
    }
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::simplify_state(const typename DMRG<MPS>::elt_t &Pk)
  {
    if (Q_values.is_empty()) {
      return reshape(Pk, Pk.size());
    } else {
      return reshape(Pk,full_size_)(range(valid_cells_));
    }
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::simplify_operator(const elt_t &H)
  {
    if (Q_values.is_empty()) {
      return H;
    } else {
      return elt_t(H(range(valid_cells_), range(valid_cells_)));
    }
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::reconstruct_state(const typename DMRG<MPS>::elt_t &Psimple)
  {
    if (Q_values.is_empty()) {
      return Psimple;
    } else {
      elt_t output(full_size_);
      output.fill_with_zeros();
      output.at(range(valid_cells_)) = Psimple;
      return output;
    }
  }


  /**********************************************************************
   * TWO-SITES DMRG ALGORITHM
   */

  template<class MPS>
  double
  DMRG<MPS>::minimize_two_sites(MPS &P, index k, int dk, index Dmax)
  {
    /* We must find out the dimensions of the problem, and also a suitable initial
     * condition for the minimization. This we do by inspecting the wavefunction
     * from the previous step.
     */
    index a1,i1,b1,j1,c1;
    elt_t Pi = P[k];
    elt_t Pj = P[k+1];
    Pi.get_dimensions(&a1, &i1, &b1);
    Pj.get_dimensions(&b1, &j1, &c1);

    index L1 = a1*i1, L2 = j1*c1;
    Pi = reshape(fold(Pi, -1, Pj, 0), L1*L2);
    Pj = elt_t();

    /* We now build the components of the Hamiltonian:
     *		Op = Opli + Opij + Opjr
     * which are the Hamiltonian for the left and right block (Opli,Opjr)
     * which now include the extra sites "i" and "j".
     */
    elt_t Opli = block_site_interaction_left(P, k);
    elt_t Opjr = block_site_interaction_right(P, k+1);
    elt_t Opij = interaction(k)
      + kron2(elt_t::eye(i1), local_term(k+1))
      + kron2(local_term(k), elt_t::eye(j1));
    /*
     * We might need to project orthogonally to a given state.
     */
    prepare_simplifier(k, Pi);
    Pi = simplify_state(Pi);
    elt_t V = projector_twosites(Pi, k);
    bool project = !V.is_empty();
    if (project) {
      Pi = Pi - fold(foldc(V,0, Pi,0),0, V,-1);
      Pi = Pi / norm2(Pi);
    }
    /*
     * Now we find the minimal energy and optimal projector.  We use an
     * iterative algorithm that does not require us to build explicitely the
     * matrix.  NOTE: For small sizes, our iterative algorithm (ARPACK) fails
     * and we have to resort to a full diagonalization.
     */
    elt_t aux;
    index smallL = Pi.size();
    index neig = std::max<int>(1, neigenvalues);
    if (Q_values.size()) {
      if (smallL <= 10) {
	//cout << "Heff=\n"; show_matrix(std::cout, kron2(Opli, elt_t::eye(j1*c1)) + kron2(elt_t::eye(a1*i1), Opjr));
	elt_t Heff_full = simplify_operator(kron2(elt_t::eye(a1),
						  kron2(Opij, elt_t::eye(c1)))
					    + direct_sum(Opli, Opjr));
	if (project) {
	  V = elt_t::eye(V.rows()) - mmult(V, adjoint(V));
	  Heff_full = mmult(adjoint(V), mmult(Heff_full, V));
	}
	sparse_t Heff(Heff_full);
	aux = eigs(Heff, linalg::SmallestAlgebraic, neig, &Pi, Pi.begin());
      } else {
	sparse_t Sli(Opli);
	sparse_t Sjr(Opjr);
	linalg::Arpack<elt_t> eigs(smallL, linalg::SmallestAlgebraic, neig);
	eigs.set_start_vector(Pi.begin());
	eigs.set_maxiter(smallL*2);
	Opij = reshape(Opij, i1,j1,i1*j1);
	while (eigs.update() < eigs.Finished) {
	  Pj = reshape(reconstruct_state(eigs.get_x()), L1,L2);
	  Pi = mmult(Sli, Pj) + mmult(Pj, Sjr) +
            reshape(foldin(Opij, -1, reshape(Pj, a1,i1*j1,c1), 1), L1,L2);
	  Pi = simplify_state(Pi);
	  if (project) {
	    Pi = Pi - fold(foldc(V,0, Pi,0),0, V,-1);
	  }
	  eigs.set_y(Pi);
	  Pi = elt_t();
	}
	if (eigs.get_status() != eigs.Finished && eigs.get_status() != eigs.TooManyIterations) {
	  std::cout << "DMRG: Diagonalization routine did not converge.\n"
		    << eigs.error_message();
	  error = true;
	  return 0.0;
	}
	aux = eigs.get_data(&Pi);
	//std::cout << "E4=" << aux[0] << '\n';
      }
    } else {
      if (smallL <= 10) {
	elt_t Heff =
	  simplify_operator(kron2(elt_t::eye(a1),
				  kron2(Opij, elt_t::eye(c1)))
			    + direct_sum(Opli, Opjr));
	if (project) {
	  V = elt_t::eye(V.rows()) - mmult(V, adjoint(V));
	  Heff = mmult(adjoint(V), mmult(Heff, V));
	}
	aux = eigs(Heff, linalg::SmallestAlgebraic, neig, &Pi, Pi.begin());
      } else {
	linalg::Arpack<elt_t> eigs(smallL, linalg::SmallestAlgebraic, neig);
	eigs.set_start_vector(Pi.begin());
	eigs.set_maxiter(smallL*2);
	Opij = reshape(Opij, i1,j1,i1*j1);
	while (eigs.update() < eigs.Finished) {
	  Pj = reshape(reconstruct_state(eigs.get_x()), L1,L2);
	  Pi = mmult(Opli, Pj) + mmult(Pj, Opjr) +
            reshape(foldin(Opij, -1, reshape(Pj, a1,i1*j1,c1), 1), L1,L2);
	  Pi = simplify_state(Pi);
	  if (project) {
	    Pi = Pi - fold(foldc(V,0, Pi,0),0, V,-1);
	  }
	  eigs.set_y(Pi);
	}
	if (eigs.get_status() != eigs.Finished && eigs.get_status() != eigs.TooManyIterations) {
	  std::cout << "DMRG: Diagonalization routine did not converge.\n"
		    << eigs.error_message();
	  error = true;
	  return 0.0;
	}
	aux = eigs.get_data(&Pi);
      }
    }
    if (neig > 1) Pi = Pi(range(), range(0));
    Pi = reconstruct_state(Pi);
    /*
     * Since the projector that we obtained spans two sites, we have to split
     * it, ensuring that we remain below the desired dimension Dmax.
     */
    set_canonical_2_sites(P, Pi, svd_tolerance, Dmax,
                          false /* Do not canonicalize the tensor, since we
                                 * are going to change it soon */);
    /*
     * And finally we update the matrices.
     */
    if (dk > 0) {
      update_matrices_left(P, k);
    } else {
      update_matrices_right(P, k+1);
    }
    eigenvalues = real(aux);
    return eigenvalues[0];
  }

  /**********************************************************************
   * COMMON LOOP
   */

  template<class MPS>
  void
  DMRG<MPS>::show_state_info(const MPS &P, index iter,
                             index k, double newE)
  {
    std::cout << "k=" << k << "; iteration=" << iter << "; E=" << newE
	      << "; E'=" << expected(P, *H_, 0);
  }

  template<class MPS>
  double
  DMRG<MPS>::minimize(MPS *Pptr, index Dmax, double E)
  {
    MPS &P = *Pptr;
    int dk=+1, k0, kN;

    //SpecialVar<bool> old_accurate_svd(accurate_svd, true);

    tic();
    P = canonical_form(P);

    init_matrices(P, 0, Dmax > 0);

    int failures = 0;
    for (index L = size(), iter = 0; iter < sweeps; iter++, dk=-dk) {
      double newE;
      /*
       * We sweep from the left to the right or viceversa. We also try to
       * avoid minimizing twice the same site, and for that reason the backward
       * sweeps begin one site to the left of the last optimized site.
       * NOTE: If we use a two-sites algorithm, we have to be careful because
       * the pathological case of L=2 leads to an iteration being empty [1]
       */
      if (Dmax) {
	if (dk > 0) {
	  k0 = 0; kN = L-1; dk = +1;
	} else if (L < 3) {
	  continue;
	} else {
	  k0 = L-2; kN = -1; dk = -1;
	}
      } else {
	if (dk > 0) {
	  k0 = 0; kN = L; dk = +1;
	} else {
	  k0 = L-1; kN = 0; dk = -1;
	}
      }
      for (int k = k0; k != kN; k += dk) {
	error = false;
	if (Dmax) {
	  newE = DMRG::minimize_two_sites(P, k, dk, Dmax);
	} else {
	  newE = DMRG::minimize_single_site(P, k, dk);
	}
	if (error) {
	  P = canonical_form(P);
	  return E;
	}
	if (debug > 2) {
	  show_state_info(P, iter, k, newE);
	}
      }
      if (debug) {
	if (debug > 1) {
	  show_state_info(P, iter, 0, newE);
	} else {
	  std::cout << "k=" << 0 << "; iteration=" << iter << "; E=" << newE << "; ";
	}
	std::cout << "dE=" << newE - E << '\n' << std::flush;
      }
      /*
       * Check the convergence by seeing how much the energy changed between
       * iterations.
       */
      if (iter) {
	if (abs(newE-E) < tolerance) {
	  if (debug) {
	    std::cout << "Reached tolerance dE=" << newE-E
		      << "<=" << tolerance << '\n' << std::flush;
	  }
	  E = newE;
	  break;
	}
	if ((newE - E) > 1e-14*abs(newE)) {
	  if (debug) {
	    std::cout << "Energy does not decrease!\n" << std::flush;
	  }
	  if (failures >= allow_E_growth) {
	    E = newE;
	    break;
	  }
	  failures++;
	}
      }
      E = newE;
    }
    if (debug) {
      std::cout << "Time used: " << toc() << "s\n";
    }
    return E;
  }


  /**********************************************************************
   * BUILDING THE EFFECTIVE HAMILTONIANS
   */

  /*
   * The Hamiltonian class always returns complex operators. We can
   * fix this by having a specialization depending on the DMRG class.
   */
  static inline const RTensor to_tensor(const DMRG<RMPS> &dmrg, const CTensor &op)
  {
    return real(op);
  }

  static inline const CTensor to_tensor(const DMRG<CMPS> &dmrg, const CTensor &op)
  {
    return op;
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t DMRG<MPS>::interaction(index k) const
  {
    return to_tensor(*this, H_->interaction(k));
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t DMRG<MPS>::interaction_left(index k, index m) const
  {
    return to_tensor(*this, H_->interaction_left(k, m));
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t DMRG<MPS>::interaction_right(index k, index m) const
  {
    return to_tensor(*this, H_->interaction_right(k, m));
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t DMRG<MPS>::local_term(index k) const
  {
    return to_tensor(*this, H_->local_term(k));
  }

  template<class MPS>
  index DMRG<MPS>::interaction_depth(index k) const
  {
    return H_->interaction_depth(k);
  }

  template<class MPS>
  void
  DMRG<MPS>::init_matrices(const MPS &P, index k0, bool also_Q)
  {
    if (also_Q) {
      Ql_ = mps_vector_t(Q_values.size(), MPS(size()));
      Qr_ = Ql_;
    } else {
      Ql_ = mps_vector_t();
      Qr_ = Ql_;
    }

    Proj_ = P0_;

    for (index i = 0; i < k0; i++) {
      update_matrices_left(P, i);
    }
    for (index i = size(); i-- > k0;) {
      update_matrices_right(P, i);
    }
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_vector_t
  DMRG<MPS>::compute_interactions_right(const MPS &P, index k) const
  {
    assert(k+1 < size());

    elt_t Pk = P[k+1];
    index a1,j1,b1;
    Pk.get_dimensions(&a1, &j1, &b1);

    index l = interaction_depth(k);
    elt_vector_t output(l);
    for (index m = 0; m < l; m++) {
      // Pk'(a1,j1,b1) O(j1,j2) Pk(a2,j2,b1) -> aux(a1,a2)
      output.at(m) = foldc(reshape(Pk,a1,j1*b1),-1,
			   reshape(foldin(interaction_right(k,m),-1, Pk,1),
				   a1,j1*b1), -1);
    }
    return output;
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_vector_t
  DMRG<MPS>::compute_interactions_left(const MPS &P, index k) const
  {
    assert(k >= 1);

    elt_t Pk = P[k-1];
    index a1,j1,b1;
    Pk.get_dimensions(&a1, &j1, &b1);

    index l = interaction_depth(k-1);
    elt_vector_t output(l);
    for (index m = 0; m < l; m++) {
      // Pk'(a1,j1,b1) O(j1,j2) Pk(a1,j2,b2) -> aux(b1,b2)
      output.at(m) = foldc(reshape(Pk, a1*j1,b1), 0,
			   reshape(foldin(interaction_left(k-1,m),-1, Pk,1),
				   a1*j1,b1), 0);
    }
    return output;
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::block_site_interaction_right(const MPS &P, index k)
  {
    index a1,i1,a2;
    P[k].get_dimensions(&a1, &i1, &a2);
    elt_t Heff = elt_t::zeros(i1*a2,i1*a2);
    if (k+1 < size()) {
      elt_vector_t Heffir = compute_interactions_right(P, k);
      Heff = kron2(elt_t::eye(i1), Hr_[k+1]);
      for (index m = 0; m < Heffir.size(); m++) {
	Heff = Heff + kron2(interaction_left(k,m), Heffir[m]);
      }
    }
    return ensure_Hermitian(Heff);
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::block_site_interaction_left(const MPS &P, index k)
  {
    index a1,i1,a2;
    P[k].get_dimensions(&a1, &i1, &a2);
    elt_t Heff = elt_t::zeros(a1*i1,a1*i1);
    if (k > 0) {
      elt_vector_t Heffli = compute_interactions_left(P, k);
      Heff = kron2(Hl_[k-1], elt_t::eye(i1));
      for (index m = 0; m < Heffli.size(); m++) {
	Heff = Heff + kron2(Heffli[m], interaction_right(k - 1,m));
      }
    }
    return Heff;
  }

  /**********************************************************************
   * PROJECTORS
   */

  template<class MPS>
  index
  DMRG<MPS>::n_orth_states() const {
    return P0_.size();
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::projector(const typename DMRG<MPS>::elt_t &Pk, index k)
  {
    elt_t V;
    for (index state = 0; state < n_orth_states(); state++) {
      elt_t Qk = P0_[state][k];
      if (k > 0) {
	Qk = fold(Proj_[state][k-1], 2, Qk, 0);
      }
      if (k+1 < size()) {
	Qk = fold(Qk, -1, Proj_[state][k+1], 0);
      }
      Qk = simplify_state(Qk);
      double n = norm2(Qk);
      if (n > 1e-14) {
	Qk = Qk / n;
      } else if (n_orth_states() == 1) {
	return elt_t();
      } else {
	Qk.fill_with_zeros();
      }
      if (V.is_empty())
	V = elt_t(Qk.size(), n_orth_states());
      V.at(range(), range(state)) = Qk;
    }
    return conj(V);
  }

  template<class MPS>
  const typename DMRG<MPS>::elt_t
  DMRG<MPS>::projector_twosites(const typename DMRG<MPS>::elt_t &Pk, index k)
  {
    elt_t V;
    for (index state = 0; state < n_orth_states(); state++) {
      elt_t Qk = fold(P0_[state][k], -1, P0_[state][k+1], 0);
      if (k > 0) {
	Qk = fold(Proj_[state][k-1], 2, Qk, 0);
      }
      if (k+2 < size()) {
	Qk = fold(Qk, -1, Proj_[state][k+2], 0);
      }
      Qk = simplify_state(Qk);
      double n = norm2(Qk);
      if (n > 1e-14) {
	Qk = Qk / n;
      } else if (n_orth_states() == 1) {
	return elt_t();
      } else {
	Qk.fill_with_zeros();
      }
      if (V.is_empty())
	V = elt_t(Qk.size(), n_orth_states());
      V.at(range(), range(state)) = Qk;
    }
    return conj(V);
  }

  template<class MPS>
  void
  DMRG<MPS>::update_matrices_right(const MPS &P, index k)
  {
    elt_t Pk = P[k];

    // We have the operator Hblock(ib,i'b') expressed on the basis |i>|b>
    // and want the same expression on the basis |a>=P(a,i,b)|i>|b>.
    index a1,i1,b1;
    Pk.get_dimensions(&a1, &i1, &b1);

    // Projector on a given state
    for (index state = 0; state < n_orth_states(); state++) {
      elt_t prev = (k+1 < size())? Proj_[state][k+1] : elt_t();
      Proj_[state].at(k) = prop_matrix(prev, -1, P0_[state][k], Pk);
    }

    // Pk'(a1,[i1b1]) H([i1b1],[i2b2]) Pk(a2,[i2b2])
    Pk = reshape(Pk, a1,i1*b1);
    elt_t Hblock = block_site_interaction_right(P, k)
      + kron2(local_term(k), elt_t::eye(b1));
    Hblock = ensure_Hermitian(Hblock);
    Hr_.at(k) = fold(foldc(Pk,-1, reshape(Hblock, i1*b1,i1*b1),0), -1, Pk, -1);

    // Conserved quantities:
    index n_Q = Q_values.size();
    for (index n = 0; n < n_Q; n++) {
      elt_t prev_Q = (k+1 < size())? Qr_[n][k+1] : elt_t::zeros(1);
      prev_Q = direct_sum(elt_t(take_diag(Q_operators[n])), prev_Q);
      prev_Q = foldc(Pk, -1, scale(Pk, -1, prev_Q), -1);
      Qr_.at(n).at(k) = round(real(take_diag(prev_Q)));
      //std::cout << "Qr_(" << n << ',' << k  << ")=\n";
      //show_matrix(std::cout, Qr_[n][k]);
    }
  }

  template<class MPS>
  void
  DMRG<MPS>::update_matrices_left(const MPS &P, index k)
  {
    elt_t Pk = P[k];

    // We have the operator Hblock(bi,b'i') expressed on the basis |b>|i>
    // and want the same expression on the basis |a>=P(b,i,a)|b>|i>.
    index b1,i1,a1;
    Pk.get_dimensions(&b1, &i1, &a1);

    // Projector on a given state
    for (index state = 0; state < n_orth_states(); state++) {
      elt_t prev = (k == 0)? elt_t() : Proj_[state][k-1];
      Proj_[state].at(k) = prop_matrix(prev, +1, P0_[state][k], Pk);
    }

    // Pk'([b1i1],a1) H([b1i1],[b2i2]) Pk([b2i2],a2)
    Pk = reshape(Pk, b1*i1,a1);
    elt_t Hblock = block_site_interaction_left(P, k)
      + kron2(elt_t::eye(b1), local_term(k));
    Hblock = (Hblock + adjoint(Hblock)) / 2.0;
    Hl_.at(k) = foldc(Pk, 0, mmult(reshape(Hblock, b1*i1,b1*i1), Pk), 0);

    // Conserved quantities:
    for (index n = 0; n < Q_values.size(); n++) {
      elt_t prev_Q = (k > 0)? Ql_[n][k-1] : elt_t::zeros(1);
      prev_Q = direct_sum(prev_Q, elt_t(take_diag(Q_operators[n])));
      prev_Q = foldc(Pk, 0, scale(Pk, 0, prev_Q), 0);
      Ql_[n].at(k) = round(real(take_diag(prev_Q)));
      //std::cout << "Ql_(" << n << ',' << k << ")=\n";
      //show_matrix(std::cout, Ql_[n][k]);
    }
  }

} // namespace mps
