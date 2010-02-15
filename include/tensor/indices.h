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

    explicit Indices(index size) : Vector<index>(size) {}

    bool operator==(const Indices &other) const;
  };

  extern template class Vector<index>;

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

  /**Create a Range which only contains one index. \ref TensorSlices*/
  inline Range *range(index ndx) { return new SingleRange(ndx); }
  /**Create a Range start:end in Matlab notation. \ref TensorSlices*/
  inline Range *range(index start, index end) { return new StepRange(start, end); }
  /**Create a Range start:step:end in Matlab notation. \ref TensorSlices*/
  inline Range *range(index start, index end, index step) { return new StepRange(start, end, step); }
  /**Create a Range with the indices in the given vector. \ref TensorSlices*/
  inline Range *range(Indices i) { return new IndexRange(i); }
  /**Create a Range which covers all indices. \ref TensorSlices*/
  inline Range *range() { return new FullRange(); }

}; // namespace

/*@}*/
#endif // !TENSOR_H
