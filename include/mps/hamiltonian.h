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

#ifndef MPS_HAMILTONIAN_H
#define MPS_HAMILTONIAN_H

#include <list>
#include <vector>
#include <tensor/sparse.h>
#include <mps/mps.h>

namespace mps {

  /*!\addtogroup Hamilt Hamiltonians */
  /**@{*/
  /**Base class for 1D lattice Hamiltonians*/
  class Hamiltonian {
  public:
    virtual ~Hamiltonian();

    /**Create a copy of this object.*/
    virtual const Hamiltonian *duplicate() const = 0;

    virtual index size() const = 0;
    /**Is there interaction between the first and the last sites?*/
    virtual bool is_periodic() const = 0;
    /**Does the Hamiltonian depend on time? */
    virtual bool is_constant() const = 0;
    /**Nearest neighbor interaction between sites 'k' and 'k+1'*/
    virtual const CTensor interaction(index k, double t = 0.0) const = 0;
    virtual const CTensor interaction_left(index k, index n, double t = 0.0) const;
    virtual const CTensor interaction_right(index k, index n, double t = 0.0) const;
    virtual index interaction_depth(index k, double t = 0.0) const;
    /**Local term of the Hamiltonian on site 'k'.*/
    virtual const CTensor local_term(index k, double t = 0.0) const = 0;
    /**Dimension of the Hilbert space on the k-th site.*/
    virtual index dimension(index k) const;
    const Indices dimensions() const;

    void reshape(index new_length);
  };

  /** Expected value of a Hamiltonian over a matrix product state.*/
  double expected(const RMPS &psi, const Hamiltonian &H, double t);

  /** Expected value of a Hamiltonian over a matrix product state.*/
  double expected(const CMPS &psi, const Hamiltonian &H, double t);

  /** Create a sparse matrix using the information in Hamiltonian.*/
  const CSparse sparse_hamiltonian(const Hamiltonian &H, double t = 0.0);

  /**1D Hamiltonian, translationally invariant and constant.*/
  class TIHamiltonian: public Hamiltonian {

  public:
    TIHamiltonian(index N, const CTensor &newH12, const CTensor &newH1, bool periodic = 0);

    virtual const Hamiltonian *duplicate() const;
    virtual index size() const;
    virtual bool is_periodic() const;
    virtual bool is_constant() const;
    virtual const CTensor interaction(index k, double t) const;
    virtual const CTensor interaction_left(index k, index n, double t) const;
    virtual const CTensor interaction_right(index k, index n, double t) const;
    virtual index interaction_depth(index k, double t = 0.0) const;
    virtual const CTensor local_term(index k, double t) const;

  private:
    index size_;
    CTensor H12_;
    CTensor H1_;
    bool periodic_;
    std::vector<CTensor> H12_left_, H12_right_;
  };

  /**1D Hamiltonian, constant but with no translational invariance*/
  class ConstantHamiltonian: public Hamiltonian {

  public:
    ConstantHamiltonian(index N, bool periodic = false);

    void set_interaction(index k, const CTensor &H1, const CTensor &H2);
    void add_interaction(index k, const CTensor &H1, const CTensor &H2);
    void set_local_term(index k, const CTensor &H1);

    virtual const Hamiltonian *duplicate() const;
    virtual index size() const;
    virtual bool is_periodic() const;
    virtual bool is_constant() const;
    virtual const CTensor interaction(index k, double t) const;
    virtual const CTensor interaction_left(index k, index n, double t) const;
    virtual const CTensor interaction_right(index k, index n, double t) const;
    virtual index interaction_depth(index k, double t = 0.0) const;
    virtual const CTensor local_term(index k, double t) const;

  private:

    const CTensor compute_interaction(index k) const;

    std::vector<CTensor> H12_, H1_;
    std::vector<std::vector<CTensor> > H12_left_, H12_right_;
    bool periodic_;
  };

  void split_interaction(const CTensor &H12, std::vector<CTensor> *v1, std::vector<CTensor> *v2);

  /**@}*/

} // namespace mps

#endif /* !MPS_HAMILTONIAN_H */
