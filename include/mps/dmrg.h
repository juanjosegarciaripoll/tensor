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

#ifndef MPS_DMRG_H
#define MPS_DMRG_H

#include <mps/mps.h>
#include <mps/hamiltonian.h>

namespace mps {

  template<class MPS>
  class DMRG {
    typedef typename MPS::elt_t elt_t;
    typedef typename std::vector<elt_t> elt_vector_t;
    typedef typename std::vector<MPS> mps_vector_t;
    typedef typename tensor::Sparse<typename elt_t::elt_t> sparse_t;

  public:
    bool error;
    index sweeps;
    bool display;
    index debug;
    double tolerance;
    double svd_tolerance;
    int allow_E_growth;
    index neigenvalues;

    RTensor eigenvalues;

    RTensor Q_values;
    elt_vector_t Q_operators;

    DMRG(const Hamiltonian &H);
    virtual ~DMRG();

    void clear_orthogonality();
    void orthogonal_to(const MPS &P);
    void clear_conserved_quantities();
    void commutes_with(const elt_t &Q);

    double minimize(MPS *P, index Dmax = 0, double E = 0.0);

    index size() const { return H_->size(); }
    bool is_periodic() const { return H_->is_periodic(); }

  private:
    const Hamiltonian *H_;

    elt_vector_t Hl_;
    elt_vector_t &Hr_;

    mps_vector_t P0_, Proj_;

    mps_vector_t Ql_, Qr_;
    index full_size_;
    Indices valid_cells_;

    const elt_t interaction(index k) const;
    const elt_t interaction_left(index k, index m) const;
    const elt_t interaction_right(index k, index m) const;
    const elt_t local_term(index k) const;
    index interaction_depth(index k) const;

    void init_matrices(const MPS &P, index k0, bool also_Q);
    void update_matrices(const MPS &P, index k0, int sense);
    void update_matrices_right(const MPS &P, index k0);
    void update_matrices_left(const MPS &P, index k0);

    const elt_vector_t compute_interactions_right(const MPS &Pk, index k) const;
    const elt_vector_t compute_interactions_left(const MPS &Pk, index k) const;
    const elt_t block_site_interaction_right(const MPS &P, index k);
    const elt_t block_site_interaction_left(const MPS &P, index k);

    index n_orth_states() const;
    const elt_t projector(const elt_t &Pk, index k);
    const elt_t projector_twosites(const elt_t &Pk, index k);

    index n_constants() const;

    double minimize_single_site(MPS &P, index k, int dk);
    double minimize_two_sites(MPS &P, index k, int dk, index Dmax);

    void prepare_simplifier(index k, const elt_t &Pk);
    const elt_t simplify_state(const elt_t &Pk);
    const elt_t simplify_operator(const elt_t &H);
    const elt_t reconstruct_state(const elt_t &Psimple);

    virtual void show_state_info(const MPS &Pk, index iter, index k, double newE);

  private:
    DMRG(const DMRG &c); // Hidden, not allowed
  };

  extern template class DMRG<RMPS>;
  typedef DMRG<RMPS> RDMRG;

  extern template class DMRG<CMPS>;
  typedef DMRG<CMPS> CDMRG;

} // namespace dmrg

#endif /* !MPS_DMRG_H */
