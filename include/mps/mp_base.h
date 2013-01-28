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

#ifndef MPS_MP_BASE_H
#define MPS_MP_BASE_H

#include <vector>
#include <tensor/tensor.h>

namespace mps {

  using tensor::index;

  template<class Tensor>
  class MP {
    typedef typename std::vector<Tensor> data_type;
  public:
    typedef Tensor elt_t;
    typedef typename data_type::iterator iterator;
    typedef typename data_type::const_iterator const_iterator;

    MP() : data_() {}
    MP(size_t size) : data_(size) {}
    MP(const MP<Tensor> &other) : data_(other.data_) {}

    index size() const { return data_.size(); }
    void resize(index new_size) { data_.resize(new_size); }

    const Tensor &operator[](index n) const {
      assert((n>=0) && (n<size()));
      return data_[n];
    }
    Tensor &at(index n) {
      assert((n>=0) && (n<size()));
      return data_.at(n);
    }

    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
    iterator end() { return data_.end(); }

  private:
    data_type data_;
  };
} // namespace mps

#endif //!MPS_MP_BASE_H
