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

#ifndef MPS_MPS_H
#define MPS_MPS_H

#include <mps/tools.h>
#include <mps/rmps.h>
#include <mps/cmps.h>

namespace mps {

  using namespace tensor;

/*!\defgroup TheMPS Matrix product states

   A Matrix Product State (MPS) represents a 1D quantum state made of N sites,
   particles or spins. Each site has a tensor associated to it, so that the
   whole state may be written as a contraction of these tensors
   \f[
   |\psi\rangle = \sum_{i,\alpha,\beta} A^{i_1}_{\alpha_1\alpha_2}
   A^{i_2}_{\alpha_2\alpha_3}\cdot A^{i_N}_{\alpha_N,\alpha_1}
   |i_1,i_2,\ldots,i_N\rangle
   \f]
   Each of the tensors in this product may be different and have different sizes.
   The indices \f$i_k\f$ denote the physical state of the k-th site. The indices
   \f$\alpha_k\f$ have no direct physical interpretation. However, the larger the
   these indices can be, the more accurately we can approximate arbitrary states
   using the previous representation. Finally, in problems with open boundary
   conditions, the index \f$\alpha_1\f$ need only have size 1.

   From the point of view of the programmer, a MPS is just a collection of
   tensors. The user is allowed to put and retrieve the tensor associated to
   the k-th site using get(k) or set(k,A), where A is the new tensor. Each tensor
   typically has three indices and is organized as A(a,i,b), where \c a and \c b
   are the \f$\alpha\f$ and \f$\beta\f$ from the previous formula, and \c i
   is the phyisical degree of freedom.

   Other operations, such as orthogonalize() or orthonormalize() are related
   to the \ref Algorithms algorithms for evolution and computation of ground
   states.

   Finally, since the MPS are actually vectors, one can compute the norm(),
   a scalar product with scprod(), expected values with correlation(), or
   obtain a vector that represents the same state with to_basis().
*/

  /**Create a product state. */
  const RMPS product_state(index length, const tensor::RTensor &local_state);

  /**Create a product state. */
  const CMPS product_state(index length, const tensor::CTensor &local_state);

  /**Create a GHZ state.*/
  const RMPS ghz_state(index length, bool periodic = false);

  /**Create a cluster state.*/
  const RMPS cluster_state(index length);

  /** Apply a local operator on the given site. */
  const RMPS apply_local_operator(const RMPS &psi, const RTensor &op, index site);

  /** Apply a local operator on the given site. */
  const CMPS apply_local_operator(const CMPS &psi, const CTensor &op, index site);

  /**Convert a RMPS to a complex vector, contracting all tensors.*/
  const RTensor mps_to_vector(const RMPS &mps);

  /**Convert a CMPS to a complex vector, contracting all tensors.*/
  const CTensor mps_to_vector(const CMPS &mps);

  /**Norm of a RMPS.*/
  double norm2(const RMPS &psi);

  /**Norm of a CMPS.*/
  double norm2(const CMPS &psi);

  /**Scalar product between MPS.*/
  double scprod(const RMPS &psi1, const RMPS &psi2);

  /**Scalar product between MPS.*/
  cdouble scprod(const CMPS &psi1, const CMPS &psi2);

  /**Compute a single-site expected value.*/
  double expected(const RMPS &a, const RTensor &Op1, index k);

  /**Add the expectation values a single-site operator over the lattice.*/
  double expected(const RMPS &a, const RTensor &Op1);

  /**Compute a single-site expected value.*/
  cdouble expected(const RMPS &a, const CTensor &Op1, index k);

  /**Add the expectation values a single-site operator over the lattice.*/
  cdouble expected(const RMPS &a, const CTensor &Op1);

  /**Compute a single-site expected value.*/
  cdouble expected(const CMPS &a, const CTensor &Op1, index k);

  /**Add the expectation values a single-site operator over the lattice.*/
  cdouble expected(const CMPS &a, const CTensor &Op1);

  /**Compute a two-site correlation.*/
  double expected(const RMPS &a, const RTensor &op1, index k1, const RTensor &op2, index k2);

  /**Compute a two-site correlation.*/
  cdouble expected(const RMPS &a, const CTensor &op1, index k1, const CTensor &op2, index k2);

  /**Compute a two-site correlation.*/
  cdouble expected(const CMPS &a, const CTensor &op1, index k1, const CTensor &op2, index k2);

  /**Store a tensor in a matrix product state in the canonical form.*/
  void set_canonical(RMPS &psi, index site, const RTensor &A, int sense, bool truncate = true);

  /**Store a tensor in a matrix product state in the canonical form.*/
  void set_canonical(CMPS &psi, index site, const CTensor &A, int sense, bool truncate = true);

  /**Rewrite a RMPS in canonical form.*/
  const RMPS canonical_form(const RMPS &psi, int sense = -1);

  /**Rewrite a CMPS in canonical form.*/
  const CMPS canonical_form(const CMPS &psi, int sense = -1);

  /**Rewrite a RMPS in canonical form, normalizing.*/
  const RMPS normal_form(const RMPS &psi, int sense = -1);

  /**Rewrite a CMPS in canonical form, normalizing.*/
  const CMPS normal_form(const CMPS &psi, int sense = -1);

  /** Update an MPS with a tensor that spans two sites, (site,site+1). Dmax is
   * the maximum bond dimension that is used. Actually, tol and Dmax are the
   * arguments to where_to_truncate. */
  void set_canonical_2_sites(RMPS &P, const RTensor Pij, index site, int sense,
                             index Dmax = 0, double tol = -1, bool normalize = true);

  /** Update an MPS with a tensor that spans two sites, (site,site+1). Dmax is
   * the maximum bond dimension that is used. Actually, tol and Dmax are the
   * arguments to where_to_truncate. */
  void set_canonical_2_sites(CMPS &P, const CTensor Pij, index site, int sense,
                             index Dmax = 0, double tol = -1, bool normalize = true);

}

#endif /* !TENSOR_MPS_H */
