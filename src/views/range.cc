// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <iostream>
#include <tensor/indices.h>

namespace tensor {

  Range::Range()
    : base_(0), limit_(0), factor_(1)
  {
  }

  Range::~Range()
  {
  }

  index
  Range::pop()
  {
    return nomore();
  }

  void
  Range::set_factor(index factor)
  {
    factor_ = factor;
  }

  void
  Range::set_limit(index limit)
  {
    assert(limit >= 0);
    limit_ = limit;
  }

  index
  Range::size() const
  {
    return 0;
  }

  void
  Range::reset()
  {
  }

  /************************************************************
   * ALL-INDEX RANGE
   */

  FullRange::FullRange()
    : Range(), counter_(0), counter_end_(0)
  {
  }

  index
  FullRange::pop()
  {
    if (counter_ >= counter_end_) {
      return nomore();
    } else {
      index output = counter_;
      counter_ += get_factor();
      return output;
    }
  }

  void
  FullRange::set_factor(index factor)
  {
    index old = get_factor();
    counter_ = 0;
    counter_end_ = get_limit() * factor;
    Range::set_factor(factor);
  }

  void
  FullRange::set_limit(index limit)
  {
    counter_end_ = limit * get_factor();
    Range::set_limit(limit);
  }

  index
  FullRange::size() const
  {
    return get_limit();
  }

  void
  FullRange::reset()
  {
    counter_ = 0;
  }

  /************************************************************
   * SINGLE INDEX RANGE
   */
  SingleRange::SingleRange(index ndx)
    : Range(), ndx_(ndx), counter_(ndx)
  {
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
  SingleRange::set_limit(index limit)
  {
    assert(limit > ndx_);
    Range::set_limit(limit);
  }

  index
  SingleRange::size() const
  {
    return 1;
  }

  void
  SingleRange::reset()
  {
    counter_ = 0;
  }

  /************************************************************
   * EQUALLY SPACED INDICES RANGE
   */

  StepRange::StepRange(index start, index end, index step) :
    Range(), start_(start), end_(end), step_(step), ndx_(0)
  {
  }

  index
  StepRange::pop()
  {
    if (ndx_ > end_) {
      return nomore();
    } else {
      index output = ndx_;
      ndx_ += step_;
      return output;
    }
  }

  void
  StepRange::set_factor(index factor)
  {
    index old = get_factor();
    end_ = (end_ / old) * factor;
    step_ = (step_ / old) * factor;
    ndx_ = start_ = (start_ / old) * factor;
    Range::set_factor(factor);
  }

  void
  StepRange::set_limit(index limit)
  {
    assert(limit > end_);
    Range::set_limit(limit);
  }

  index
  StepRange::size() const
  {
    return (end_ - start_) / step_ + 1;
  }

  void
  StepRange::reset()
  {
    ndx_ = start_;
  }

  /************************************************************
   * RANGE WITH A VECTOR OF INDICES
   */
  IndexRange::IndexRange(const Indices &i)
    : Range(), indices_(i), counter_(0)
  {
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
    counter_ = 0;
    Range::set_factor(factor);
  }

  void
  IndexRange::set_limit(index limit)
  {
#ifndef NDEBUG
    for (Indices::const_iterator it = indices_.begin(); it != indices_.end(); it++) {
      assert(*it <= limit);
    }
#endif
    Range::set_limit(limit);
  }

  index
  IndexRange::size() const
  {
    return indices_.size();
  }

  void
  IndexRange::reset()
  {
    counter_ = 0;
  }

  /************************************************************
   * TENSOR PRODUCT RANGE
   */
  ProductRange::ProductRange(Range *r1, Range *r2) : r1_(r1), r2_(r2)
  {
    r2_->set_factor(r1_->get_factor() * r1_->get_limit());
  }

  ProductRange::~ProductRange()
  {
    delete r1_;
    delete r2_;
  }

  index
  ProductRange::pop()
  {
    index out = base_;
    do {
      if (out == nomore())
        return out;
      index n = r1_->pop();
      if (n != nomore())
        return n + out;
      out = base_ = r2_->pop();
      r1_->reset();
    } while (1);
  }

  void
  ProductRange::reset()
  {
    r1_->reset();
    r2_->reset();
    base_ = r2_->pop();
  }

  void
  ProductRange::set_factor(index factor)
  {
    r1_->set_factor(factor);
    r2_->set_factor(factor * r1_->get_limit());
  }

  void
  ProductRange::set_limit(index limit)
  {
    assert(0);
  }

  index
  ProductRange::size() const
  {
    return r1_->size() * r2_->size();
  }

  Range *product(Range *r1, Range *r2)
  {
    return new ProductRange(r1, r2);
  }

} // namespace tensor
