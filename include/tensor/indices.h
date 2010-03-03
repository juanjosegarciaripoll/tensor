// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_INDICES_H
#define TENSOR_INDICES_H

#include <list>
#include <vector>
#include <tensor/vector.h>

/*!\addtogroup Tensors */
/*@{*/
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
    Indices(const Vector<index> &v) : Vector<index>(v) {}
    explicit Indices(index size) : Vector<index>(size) {}

    bool operator==(const Indices &other) const;
  };

  extern template class Vector<index>;


  template<typename elt_t, size_t n>
  class StaticVector {
  public:
    StaticVector(const StaticVector<elt_t,n-1> &other, elt_t x) :
      inner(other), extra(x)
    {};
    operator Vector<elt_t>() const {
      Vector<elt_t> output(n);
      push(output.begin());
      return output;
    }
    void push(elt_t *v) const {
      inner.push(v);
      v[n-1] = extra;
    }
  protected:
    StaticVector<elt_t,n-1> inner;
    elt_t extra;
  };

  template<typename elt_t>
  class StaticVector<elt_t,1> {
  public:
    StaticVector(elt_t x) :
      extra(x)
    {};
    operator Vector<elt_t>() const {
      Vector<elt_t> output(1);
      push(output.begin());
      return output;
    }
    void push(elt_t *v) const {
      v[0] = extra;
    }
  private:
    elt_t extra;
  };

  const StaticVector<index,1>
  operator>>(const ListGenerator &g, const int x) {
    return StaticVector<index,1>((index)x);
  }

  template<size_t n>
  const StaticVector<index,n+1>
  operator>>(const StaticVector<index,n> &g, const int x) {
    return StaticVector<index,n+1>(g,(index)x);
  }

  template<typename elt_t>
  StaticVector<elt_t,1> operator>>(const ListGenerator &g, const elt_t x) {
    return StaticVector<elt_t,1>(x);
  }

  template<typename t1, typename t2, size_t n>
  StaticVector<t1,n+1> operator>>(const StaticVector<t1,n> &g, const t2 x) {
    return StaticVector<t1,n+1>(g,x);
  }

  //////////////////////////////////////////////////////////////////////
  // RANGE OF INTEGERS
  //

  /** Range of indices. This class should never be used by public functions, but
      only as the output of the function range() and only to access segments of
      a tensors, as in
      \code
      b = a(range(1,2),range())
      \endcode
  */
  class Range {
  public:
    Range();
    virtual ~Range();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void set_limit(index new_limit);
    virtual index size() const;
    virtual void reset();
    index nomore() const { return ~(index)0; }
    index get_offset() const { return base_; }
    index get_limit() const { return limit_; }
    index get_factor() const { return factor_; }
    void set_offset(index new_base) { base_ = new_base; }
  private:
    index base_, limit_, factor_;
  };

  class FullRange : public Range {
  public:
    FullRange();
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void set_limit(index new_limit);
    virtual index size() const;
    virtual void reset();
  private:
    index counter_, counter_end_;
  };

  class StepRange : public Range {
  public:
    StepRange(index start, index end, index step = 1);
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void set_limit(index new_limit);
    virtual index size() const;
    virtual void reset();
  private:
    index ndx_, start_, end_, step_;
  };

  class SingleRange : public Range {
  public:
    SingleRange(index ndx);
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void set_limit(index new_limit);
    virtual index size() const;
    virtual void reset();
  private:
    index ndx_, counter_;
  };

  class IndexRange : public Range {
  public:
    IndexRange(const Indices &i);
    virtual index pop();
    virtual void set_factor(index new_factor);
    virtual void set_limit(index new_limit);
    virtual index size() const;
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
    virtual void set_limit(index new_limit);
    virtual index size() const;
    virtual void reset();
  private:
    Range *r1_, *r2_;
    index base_;
  };

  /**Create a Range which only contains one index. \sa \ref sec_tensor_view*/
  inline Range *range(index ndx) { return new SingleRange(ndx); }
  /**Create a Range start:end (Matlab notation). \sa \ref sec_tensor_view*/
  inline Range *range(index start, index end) { return new StepRange(start, end); }
  /**Create a Range start:step:end (Matlab notation). \sa \ref sec_tensor_view*/
  inline Range *range(index start, index end, index step) { return new StepRange(start, end, step); }
  /**Create a Range with the give set of indices. \sa \ref sec_tensor_view*/
  inline Range *range(Indices i) { return new IndexRange(i); }
  /**Create a Range which covers all indices. \ref sec_tensor_view*/
  inline Range *range() { return new FullRange(); }

}; // namespace

/*@}*/
#endif // !TENSOR_H
