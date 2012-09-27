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
        const Tensor &newB, const Tensor &newlB);

  Tensor reduced_density_matrix() const;

  elt_t expected_value(const Tensor &Op, int site = 0) const;
  elt_t expected_value(const Tensor &Op1, const Tensor &Op2,
                       tensor::index separation = 0, int site = 0) const;
  elt_t string_order(const Tensor &Opfirst, const Tensor &Opmiddle,
                     const Tensor &Oplast, tensor::index separation) const;

  void apply_operator(const Tensor &U, double tolerance = -1);

  double evolve_itime(const Tensor &H12, double dt, tensor::index nsteps,
                      double tolerance = -1, tensor::index deltan = 1);

  double entropy() const;

private:
  iTEBD();
  iTEBD(const iTEBD &other);
  Tensor A, lA, B, lB;
};

}

#endif // MPS_QUANTUM_H
