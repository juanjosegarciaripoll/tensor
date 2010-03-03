// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

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
  Tensor<elt_t>::operator()(Range *r) const
  {
    Indices dims(1);
    r->set_limit(dimension(0));
    dims.at(0) = r->size();
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(Range *r1, Range *r2) const
  {
    Indices dims(2);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    Range *r = product(r1, r2);
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename Tensor<elt_t>::view
  Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3) const
  {
    Indices dims(3);
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
  Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3, Range *r4) const
  {
    Indices dims(4);
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
  Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3, Range *r4, Range *r5) const
  {
    Indices dims(5);
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
    return view(*this, dims, r);
  }

  template<typename elt_t> const typename  Tensor<elt_t>::view
  Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3,
                            Range *r4, Range *r5, Range *r6) const
  {
    Indices dims(6);
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
  Tensor<elt_t>::at(Range *r)
  {
    Indices dims(1);
    r->set_limit(dimension(0));
    dims.at(0) = r->size();
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(Range *r1, Range *r2)
  {
    Indices dims(2);
    r1->set_limit(dimension(0));
    r2->set_limit(dimension(1));
    dims.at(0) = r1->size();
    dims.at(1) = r2->size();
    Range *r = product(r1, r2);
    return mutable_view(*this, dims, r);
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view
  Tensor<elt_t>::at(Range *r1, Range *r2, Range *r3)
  {
    Indices dims(3);
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
  Tensor<elt_t>::at(Range *r1, Range *r2, Range *r3, Range *r4)
  {
    Indices dims(4);
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
  Tensor<elt_t>::at(Range *r1, Range *r2, Range *r3, Range *r4, Range *r5)
  {
    Indices dims(5);
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
  Tensor<elt_t>::at(Range *r1, Range *r2, Range *r3,
                    Range *r4, Range *r5, Range *r6)
  {
    Indices dims(6);
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

  template<typename elt_t> typename Tensor<elt_t>::mutable_view &
  Tensor<elt_t>::mutable_view::operator=(const Tensor<elt_t>::mutable_view &t)
  {
    Range *r1 = ranges_;
    Range *r2 = t.ranges_;
    r1->reset();
    r2->reset();
    for (index i, j;
         (i = r1->pop(), j = r2->pop(), i != r1->nomore() && j != r2->nomore()); ) {
      data_.at(i) = t.data_[j];
    }
    return *this;
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view &
  Tensor<elt_t>::mutable_view::operator=(const Tensor<elt_t> &t)
  {
    ranges_->reset();
    for (index i = 0, j; (j = ranges_->pop()) != ranges_->nomore(); i++) {
      data_.at(j) = t[i];
    }
    return *this;
  }

  template<typename elt_t> typename Tensor<elt_t>::mutable_view &
  Tensor<elt_t>::mutable_view::operator=(elt_t v)
  {
    ranges_->reset();
    for (index i; (i = ranges_->pop()) != ranges_->nomore(); ) {
      data_.at(i) = v;
    }
    return *this;
  }



} // namespace tensor

