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

#ifndef MPS_ITEBD_H
#define MPS_ITEBD_H

#include <tensor/tensor.h>

/*!\addtogroup MPS*/
/** Namespace for classes and algorithms related to Matrix Product States, DMRG and similar methods. */
namespace mps {

using namespace tensor;

/*!\addtogroup MPS*/
/* @{ */
/**An infinite Matrix Product State with translational invariance
   but using two tensors: one for odd and one for even sites. This
   algorithm follows the iTBD implementation sketched by R. Orus
   and G. Vidal in <a href='http://arxiv.org/abs/0711.3960'>
   Phys. Rev. B 78, 155117 (2008)</a>
*/
template<class Tensor>
class iTEBD {
public:
  typedef typename Tensor::elt_t elt_t;
  typedef Tensor t;

  /** Create an random iTEBD state with given physical dimensions. */
  explicit iTEBD(tensor::index dimension);

  /** Create a product state iTEBD. */
  iTEBD(const Tensor &newA);

  /** Create a product state iTEBD with two wavefunctions. */
  iTEBD(const Tensor &newA, const Tensor &newB);

  /** Create an iTEBD with \f$\Gamma\f$ and \f$\lambda\f$ matrices. */
  iTEBD(const Tensor &newA, const Tensor &newlA,
        const Tensor &newB, const Tensor &newlB,
        bool canonical = false);

  /** Is this iTEBD state in canonical form? */
  bool is_canonical() const { return canonical_; }

  /** Construct a new state after acting on 'odd' or 'even' pair of sites with the two-site operator U. */
  const iTEBD<Tensor> apply_operator(const Tensor &U, int odd = 0, double tolerance = -1, tensor::index max_dim = 0) const;

  /** Return a new state which is in canonical form. */
  const iTEBD<Tensor> canonical_form() const;

  /** Estimate the entanglement entropy associated to splitting the state around 'site' */
  double entropy(int site) const;

  /** Schmidt numbers. */
  const Tensor schmidt(int site = 0) const;

  /** Average entropy for even and odd sites. */
  double entropy() const { return entropy(0) + entropy(1); }

  /* Implicit coercion to other types */
  template<class tensor>
  operator const iTEBD<tensor>() const {
    return iTEBD<tensor>(A_, lA_, B_, lB_, canonical_);
  }

  /** Return the \f$\Gamma\f$ matrix for the lattice 'site'. */
  const Tensor &matrix(int site) const {
    return (site & 1)? B_ : A_;
  }

  /** Fold a \f$\Gamma\f$ matrix with its neighboring vector \f$\lambda\f$ . */
  const Tensor &combined_matrix(int site) const {
    return (site & 1)? BlB_ : AlA_;
  }

  /** Return the vector \f$\lambda\f$ to the left of this site. */
  const Tensor &left_vector(int site) const {
    return (site & 1)? lA_ : lB_;
  }

  /** Return the vector \f$\lambda\f$ to the right of this site. */
  const Tensor &right_vector(int site) const {
    return (site & 1)? lB_ : lA_;
  }

  /** Physical dimension of the given site. */
  tensor::index site_dimension(int site) const {
    return matrix(site).dimension(1);
  }

  /** Return dimension of the MPS to the left of this site. */
  tensor::index left_dimension(int site) const {
    return matrix(site).dimension(0);
  }

  /** Return dimension of the MPS to the right of this site. */
  tensor::index right_dimension(int site) const {
    return matrix(site).dimension(2);
  }

  const Tensor left_boundary(int site) const {
    return diag(left_vector(site) * left_vector(site));
  }

  const Tensor right_boundary(int site) const {
    return diag(right_vector(site) * right_vector(site));
  }

private:
  /* Avoid initializing empty iTEBD states. */
  iTEBD();

  /* Given a two-site wavefunction, split it. */
  iTEBD(const Tensor &AB, const Tensor &lAB, double tolerance, tensor::index max_dim);

  Tensor A_, B_;
  Tensor lA_, lB_;
  Tensor AlA_, BlB_;
  bool canonical_;
};

  typedef iTEBD<RTensor> RiTEBD;
  typedef iTEBD<CTensor> CiTEBD;

  /** Infinite long, one-dimensional GHZ state. */
  const RiTEBD infinite_ghz_state();

  /** Infinitely long, one-dimensional cluster state. */
  const RiTEBD infinite_cluster_state();

  /** Infinitely long, one-dimensional AKLT state. */
  const RiTEBD infinite_aklt_state();

  /** Expected value of an operator acting on 'site'. */
  double expected(const RiTEBD &psi, const RTensor &Op, int site = 0);

  /** Expected value of two operators, acting on sites '0' and '1'. */
  double expected(const RiTEBD &psi, const RTensor &Op1, const RTensor &Op2);

  //** Expected value of two operators, acting on sites 'i' and 'j'. */
  double expected(const RiTEBD &psi, const RTensor &Op1, int i,
                  const RTensor &Op2, int j);

  /** String order parameter between sites 'i' and 'j', both included. */
  double string_order(const RiTEBD &psi, const RTensor &Opi, int i,
                      const RTensor &Opmiddle,
                      const RTensor &Opj, int j);

  /** Expected value of the two-site operator Op12 acting on 'site' and 'site+1'. */
  double expected12(const RiTEBD &psi, const RTensor &Op12, int site = 0);

  /** Expected value of the two-site operator Op12 acting on 'site' and 'site+1'. */
  double energy(const RiTEBD &psi, const RTensor &Op12);


  /** Expected value of an operator acting on 'site'. */
  cdouble expected(const CiTEBD &psi, const CTensor &Op, int site = 0);

  /** Expected value of two operators, acting on sites '0' and '1'. */
  cdouble expected(const CiTEBD &psi, const CTensor &Op1, const CTensor &Op2);

  //** Expected value of two operators, acting on sites 'i' and 'j'. */
  cdouble expected(const CiTEBD &psi, const CTensor &Op1, int i,
                  const CTensor &Op2, int j);

  /** String order parameter between sites 'i' and 'j', both included. */
  cdouble string_order(const CiTEBD &psi, const CTensor &Opi, int i,
                      const CTensor &Opmiddle,
                      const CTensor &Opj, int j);

  /** Expected value of the two-site operator Op12 acting on 'site' and 'site+1'. */
  cdouble expected12(const CiTEBD &psi, const CTensor &Op12, int site = 0);

  /** Expected value of the two-site operator Op12 acting on 'site' and 'site+1'. */
  double energy(const CiTEBD &psi, const CTensor &Op12);


/** Evolve an iTEBD in imaginary time, using the local Hamiltonian \a H12 on
 state \a psi. Given a Hamiltonian which is a composition of local Hamiltonians
 acting on pairs of sites, \f$H=\sum_{i} H_{i,i+1}\f$, we evolve the iTEBD state
 \a psi using \a nsteps repetitions of the elementary time interval \a dt. \a
 tolerance and \a max_dim determine the truncation strategy of the state, while
 \a deltan (if nonzero) instructs the program to report the properties of the
 state on the standard text output. */
template<class Tensor>
const iTEBD<Tensor> evolve_itime(iTEBD<Tensor> psi, const Tensor &H12, double dt,
                                 tensor::index nsteps, double tolerance = -1,
                                 tensor::index max_dim = 0, tensor::index deltan = 1);

/* @} */

}

#endif // MPS_QUANTUM_H
