// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_INDICES_H
#define TENSOR_INDICES_H

#include <list>
#include <vector>
#include <tensor/vector.h>

namespace tensor {

  class ListGenerator  {};

  extern ListGenerator gen;

  template<typename elt_t>
  std::list<elt_t> &operator<<(std::list<elt_t> &l, const elt_t &x) {
    l.push_back(x);
    return l;
  }

  template<typename elt_t>
  std::list<elt_t> operator<<(const ListGenerator &g, const elt_t &x) {
    std::list<elt_t> output;
    output.push_back(x);
    return output;
  }

  class Indices : public Vector<index> {
  public:
    Indices() : Vector<index>() {}

    explicit Indices(index size) : Vector<index>(size) {}

    bool operator==(const Indices &other) const;
  };

  extern template class Vector<index>;

  //////////////////////////////////////////////////////////////////////
  // RANGE OF INTEGERS
  //

  class Range {
  public:
    Range(index lim = 0) : base_(0), limit_(lim), factor_(1) {}
    virtual ~Range() = 0;
    virtual index pop() = 0;
    virtual void set_factor(index new_factor) = 0;
    virtual void reset() = 0;
    index nomore() { return ~(index)0; }
    index get_offset() { return base_; }
    index get_limit() { return limit_; }
    index get_factor() { return factor_; }
    void set_offset(index new_base) { base_ = new_base; }
    void set_limit(index new_limit) { limit_ = new_limit; }
  private:
    index base_, limit_, factor_;
  };

  class FullRange : public Range {
  public:
    FullRange(index start, index end, index lim = 0) :
      Range(lim), start_(start), end_(end), ndx_(0) {}
    ~FullRange();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void reset();
  private:
    index ndx_, start_, end_;
  };

  class SingleRange : public Range {
  public:
    SingleRange(index ndx, index lim = 0) : Range(lim), ndx_(ndx), counter_(ndx) {}
    ~SingleRange();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void reset();
  private:
    index ndx_, counter_;
  };

  class IndexRange : public Range {
  public:
    IndexRange(const Indices &i, index lim = 0) : Range(lim), indices_(i), counter_(0) {}
    ~IndexRange();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void reset();
  private:
    Indices indices_;
    index counter_;
  };

  class ProductRange : public Range {
  public:
    ProductRange(Range *r1, Range *r2);
    ~ProductRange();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void reset();
  private:
    Range *r1_, *r2_;
  };

  inline Range *range(index ndx) { return new SingleRange(ndx); }
  inline Range *range(index start, index end) { return new FullRange(start, end); }
  inline Range *range(Indices i) { return new IndexRange(i); }

}; // namespace

#endif // !TENSOR_H
