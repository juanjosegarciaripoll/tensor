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

  iTEBD(const Tensor &newA);
  iTEBD(const Tensor &newA, const Tensor &newB);
  iTEBD(const Tensor &newA, const Tensor &newlA,
        const Tensor &newB, const Tensor &newlB,
        bool canonical = false);

  bool is_canonical() const { return canonical_; }

  const Tensor &combined_matrix(int site) const {
    return (site & 1)? AlA_ : BlB_;
  }

  const Tensor &left_vector(int site) const {
    return (site & 1)? lB_ : lA_;
  }

  const Tensor &right_vector(int site) const {
    return (site & 1)? lA_ : lB_;
  }

  const Tensor left_boundary(int site) const {
    return diag(left_vector(site) * left_vector(site));
  }

  tensor::index site_dimension(int site) const {
    return ((site & 1)? A_ : B_).dimension(1);
  }

  elt_t expected_value(const Tensor &Op, int site = 0) const;
  elt_t expected_value(const Tensor &Op1, const Tensor &Op2,
                       tensor::index separation = 0, int site = 0) const;
  elt_t string_order(const Tensor &Opfirst, const Tensor &Opmiddle,
                     const Tensor &Oplast, tensor::index separation,
                     int site = 0) const;

  const iTEBD<Tensor> canonical_form() const;

  double entropy() const;

private:
  iTEBD();

  Tensor A_, B_, lA_, lB_;
  Tensor AlA_, BlB_;
  bool canonical_;
};

template<class Tensor>
const iTEBD<Tensor> apply_operator(const iTEBD<Tensor> &psi, const Tensor &U, double tolerance = -1, tensor::index max_dim = 0);

template<class Tensor>
const iTEBD<Tensor> evolve_itime(const iTEBD<Tensor> &psi, const Tensor &H12, double dt, tensor::index nsteps, const iTEBD<Tensor> &psi, double tolerance = -1, tensor::index deltan = 1, tensor::index max_dim = 0);

template<class Tensor>
const Tensor reduced_density_matrix(const iTEBD<Tensor> &psi);

}

#endif // MPS_QUANTUM_H
