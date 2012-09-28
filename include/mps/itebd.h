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
/* @{ */
/**An infinite Matrix Product State with translational invariance
   but using two tensors: one for odd and one for even sites. This
   algorithm follows the iTBD implementation sketched by R. Orus
   and G. Vidal in Phys. Rev. B 78, 155117 (2008)
   \see \ref sec_mps
*/
namespace mps {

using namespace tensor;

template<class Tensor>
class iTEBD {
public:
  typedef typename Tensor::elt_t elt_t;

  /** Create an random iTEBD state with given physical dimensions. */
  explicit iTEBD(tensor::index dimension);

  /** Create a product state iTEBD. */
  iTEBD(const Tensor &newA);

  /** Create a product state iTEBD with two wavefunctions. */
  iTEBD(const Tensor &newA, const Tensor &newB);

  /** Create an iTEBD with \Gamma and \lambda matrices. */
  iTEBD(const Tensor &newA, const Tensor &newlA,
        const Tensor &newB, const Tensor &newlB,
        bool canonical = false);

  /** Is this iTEBD state in canonical form? */
  bool is_canonical() const { return canonical_; }

  /** Physical dimension of the given site. */
  tensor::index site_dimension(int site) const {
    return ((site & 1)? B_ : A_).dimension(1);
  }

  /** Expected value of an operator acting on 'site'. */
  elt_t expected_value(const Tensor &Op, int site = 0) const;

  /** Expected value of two operators, acting on 'site' and on 'site + separation'. */
  elt_t expected_value(const Tensor &Op1, const Tensor &Op2,
                       tensor::index separation = 0, int site = 0) const;

  /** String order parameter between 'site' and 'site + separation'. */
  elt_t string_order(const Tensor &Opfirst, const Tensor &Opmiddle,
                     const Tensor &Oplast, tensor::index separation,
                     int site = 0) const;

  /** Expected value of the two-site operator Op12 acting on 'site' and 'site+1'. */
  elt_t expected_value12(const Tensor &Op12, int site = 0) const;

  /** Energy of a Hamiltonian with local operator H12. */
  double energy(const Tensor &H12) const;

  /** Construct a new state after acting on 'odd' or 'even' pair of sites with the two-site operator U. */
  const iTEBD<Tensor> apply_operator(const Tensor &U, int odd = 0, double tolerance = -1, tensor::index max_dim = 0) const;

  /** Return a new state which is in canonical form. */
  const iTEBD<Tensor> canonical_form() const;

  /** Estimate the entanglement entropy associated to splitting the state around 'site' */
  double entropy(int site) const;

  /** Average entropy for even and odd sites. */
  double entropy() const { return entropy(0) + entropy(1); }

private:
  /* Avoid initializing empty iTEBD states. */
  iTEBD();

  /* Given a two-site wavefunction, split it. */
  iTEBD(const Tensor &AB, const Tensor &lAB, double tolerance, tensor::index max_dim);

  Tensor A_, B_, lA_, lB_;
  Tensor AlA_, BlB_;
  bool canonical_;

  const Tensor left_boundary(int site) const {
    return diag(left_vector(site) * left_vector(site));
  }

  const Tensor right_boundary(int site) const {
    return diag(right_vector(site) * right_vector(site));
  }

  /** Fold a \Gamma matrix with its neighboring vector \lambda . */
  const Tensor &combined_matrix(int site) const {
    return (site & 1)? BlB_ : AlA_;
  }

  /** Return the vector \lambda to the left of this site. */
  const Tensor &left_vector(int site) const {
    return (site & 1)? lA_ : lB_;
  }

  /** Return the vector \lambda to the right of this site. */
  const Tensor &right_vector(int site) const {
    return (site & 1)? lB_ : lA_;
  }
};

  /** Evolve an iTEBD in imaginary time, using the local Hamiltonian H_{12}. */
template<class Tensor>
const iTEBD<Tensor> evolve_itime(iTEBD<Tensor> psi, const Tensor &H12, double dt, tensor::index nsteps, const iTEBD<Tensor> &psi, double tolerance = -1, tensor::index max_dim = 0, tensor::index deltan = 1);

template<class Tensor>
const Tensor reduced_density_matrix(const iTEBD<Tensor> &psi);

/* @} */

}

#endif // MPS_QUANTUM_H
