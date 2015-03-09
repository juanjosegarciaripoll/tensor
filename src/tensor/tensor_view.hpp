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

#include <cassert>

/**\cond IGNORE */

namespace tensor {

  ////////////////////////////////////////////////////////////
  // CONVERT TENSOR VIEWS TO TENSORS
  //

  template<typename elt_t>
  Tensor<elt_t>::view::operator Tensor<elt_t>() const
  {
    Tensor<elt_t> t(dims_);
    typename Tensor<elt_t>::iterator it = t.begin();
    ranges_->reset();
    for (index j; (j = ranges_->pop()) != ranges_->nomore(); it++) {
      *it = data_[j];
    }
    return t;
  }

  ////////////////////////////////////////////////////////////
  // CONSTRUCT CONST TENSOR VIEWS
  //

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r) const
  {
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    Indices dims(1);
    r->set_limit(size());
    dims.at(0) = r->size();
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r1, PRange r2) const
  {
    Indices dims(2);
    assert(this->rank() == 2);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    Range *r = product(r1, r2);
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r1, PRange r2, PRange r3) const
  {
    Indices dims(3);
    assert(this->rank() == 3);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    Range *r = product(r1, product(r2, r3));
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r1, PRange r2, PRange r3, PRange r4) const
  {
    Indices dims(4);
    assert(this->rank() == 4);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    Range *r = product(r1, product(r2, product(r3, r4)));
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r1, PRange r2, PRange r3, PRange r4, PRange r5) const
  {
    Indices dims(5);
    assert(this->rank() == 5);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    r5->set_limit(dimension(4));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    dims.at(4) = r5->size();
    PRange r = product(r1, product(r2, product(r3, product(r4, r5))));
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename  Tensor<elt_t>::view
  Tensor<elt_t>::operator()(PRange r1, PRange r2, PRange r3,
                            PRange r4, PRange r5, PRange r6) const
  {
    Indices dims(6);
    assert(this->rank() == 6);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    r5->set_limit(dimension(4));
    r6->set_limit(dimension(5));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    dims.at(4) = r5->size();
    dims.at(5) = r6->size();
    Range *r = product(r1, product(r2, product(r3, product(r4, product(r5, r6)))));
    return view(*this, dims, r);
  }

  ////////////////////////////////////////////////////////////
  // CONSTRUCT MUTABLE TENSOR VIEWS
  //

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r)
  {
    Indices dims(1);
    // a(range) is valid for 1D and for ND tensors which are treated
    // as being 1D
    r->set_limit(size());
    dims.at(0) = r->size();
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r1, PRange r2)
  {
    Indices dims(2);
    assert(this->rank() == 2);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    Range *r = product(r1, r2);
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r1, PRange r2, PRange r3)
  {
    Indices dims(3);
    assert(this->rank() == 3);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    Range *r = product(r1, product(r2, r3));
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r1, PRange r2, PRange r3, PRange r4)
  {
    Indices dims(4);
    assert(this->rank() == 4);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    Range *r = product(r1, product(r2, product(r3, r4)));
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r1, PRange r2, PRange r3, PRange r4, PRange r5)
  {
    Indices dims(5);
    assert(this->rank() == 5);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    r5->set_limit(dimension(4));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    dims.at(4) = r5->size();
    Range *r = product(r1, product(r2, product(r3, product(r4, r5))));
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename  Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(PRange r1, PRange r2, PRange r3,
                    PRange r4, PRange r5, PRange r6)
  {
    Indices dims(6);
    assert(this->rank() == 6);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    r3->set_limit(dimension(2));
    r4->set_limit(dimension(3));
    r5->set_limit(dimension(4));
    r6->set_limit(dimension(5));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    dims.at(2) = r3->size();
    dims.at(3) = r4->size();
    dims.at(4) = r5->size();
    dims.at(5) = r6->size();
    Range *r = product(r1, product(r2, product(r3, product(r4, product(r5, r6)))));
    return mutable_view(*this, dims, r);
  }

  //////////////////////////////////////////////////////////////////////
  // ASSIGN TO MUTABLE MUTABLE_VIEWS
  //

  template<typename elt_t> void
  Tensor<elt_t>::mutable_view::operator=
  (const typename Tensor<elt_t>::view &t)
  {
    //assert(verify_tensor_dimensions_match(dims_, t.dims_));
    assert(dims_.total_size() == t.dims_.total_size());
    Range *r1 = ranges_;
    Range *r2 = t.ranges_;
    r1->reset();
    r2->reset();
    for (index i, j;
         (i = r1->pop(), j = r2->pop(), i != r1->nomore() && j != r2->nomore()); ) {
      data_.at(i) = t.data_[j];
    }
  }

  template<typename elt_t> void
  Tensor<elt_t>::mutable_view::operator=(const Tensor<elt_t> &t)
  {
    //assert(verify_tensor_dimensions_match(dims_, t.dimensions()));
    assert(dims_.total_size() == t.dims_.total_size());
    ranges_->reset();
    for (index i = 0, j; (j = ranges_->pop()) != ranges_->nomore(); i++) {
      data_.at(j) = t[i];
    }
  }

  template<typename elt_t> void
  Tensor<elt_t>::mutable_view::operator=(elt_t v)
  {
    ranges_->reset();
    for (index i; (i = ranges_->pop()) != ranges_->nomore(); ) {
      data_.at(i) = v;
    }
  }

} // namespace tensor

/**\endcond */
