// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <iostream>
#include <tensor/indices.h>

namespace tensor {

  Range::~Range()
  {
  }

  void
  Range::set_factor(index factor)
  {
    factor_ = factor;
  }

  index
  SingleRange::pop()
  {
    if (counter_)
      return nomore();
    ++counter_;
    return ndx_;
  }

  void
  SingleRange::set_factor(index factor)
  {
    index old = get_factor();
    ndx_ = (ndx_ / old) * factor;
    Range::set_factor(factor);
  }

  void
  SingleRange::reset()
  {
    counter_ = 0;
  }

  index
  FullRange::pop()
  {
    if (ndx_ == end_)
      return nomore();
    return ndx_ += get_factor();
  }

  void
  FullRange::set_factor(index factor)
  {
    index old = get_factor();
    end_ = (end_ / old) * factor;
    ndx_ = (ndx_ / old) * factor;
    start_ = (start_ / old) * factor;
    Range::set_factor(factor);
  }

  void
  FullRange::reset()
  {
    ndx_ = start_;
  }

  index
  IndexRange::pop()
  {
    if (counter_ >= indices_.size())
      return nomore();
    return indices_[counter_++];
  }

  void
  IndexRange::set_factor(index factor)
  {
    index old = get_factor();
    for (Indices::iterator it = indices_.begin(); it != indices_.end(); it++) {
      *it = (*it / old) * factor;
    }
    Range::set_factor(factor);
  }

  void
  IndexRange::reset()
  {
    counter_ = 0;
  }

  ProductRange::ProductRange(Range *r1, Range *r2) : r1_(r1), r2_(r2)
  {
    r2_->set_factor(r2_->get_factor() * r1_->get_factor() * r1_->get_limit());
  }

  ProductRange::~ProductRange()
  {
    delete r1_;
    delete r2_;
  }

  index
  ProductRange::pop()
  {
    index n = r1_->pop();
    if (n == nomore()) {
      n = r2_->pop();
      if (n == nomore()) {
        return n;
      } else {
        index n2;
        r1_->reset();
        n2 = r1_->pop();
        if (n2 == nomore())
          return n2;
        return n2 + n;
      }
    }
    return n;
  }

  void
  ProductRange::reset()
  {
    r1_->reset();
    r2_->reset();
  }

  void
  ProductRange::set_factor(index factor)
  {
    index old = r1_->get_factor();
    r2_->set_factor((r2_->get_factor() / old) * factor);
  }

} // namespace tensor
