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
    typedef enum {
      Full,
      Stepped,
      List,
      None
    } Type;

    Range() : type(Full), i0(0), i1(-1), di(1) {}

    Range(const Indices &av): type(List), v(av) {}

    Range(index ndx) : type(Stepped), i0(ndx), i1(ndx), di(1) {}

    Range(index start, index end) : type(Stepped), i0(start), i1(end), di(1) {}

    Range(index start, index end, index delta) :
	type(Stepped), i0(start), i1(end), di(delta)
    {}

    index size(index dimension) const;

  private:
    Type type;
    index i0, i1, di;
    Indices v;

    friend class RangeProduct;
    void to_offsets(Indices &i, index increment, index dimension) const;
  };

  //////////////////////////////////////////////////////////////////////
  // TENSOR PRODUCT OF RANGES
  //

  class RangeProduct {
  public:
    RangeProduct(const std::list<Range> all_ranges);
    ~RangeProduct();

    index size() const { return size_; }
    index rank() const { return ranges_.size(); }
    bool begin(const Indices &limits);
    bool reset();
    bool next();
    index offset() const;
    bool is_empty() const { return !size(); }
    const Indices &dimensions() const { return limits_; };

  private:
    typedef std::list<Range> ranges;
    ranges ranges_;
    index size_;
    std::vector<Indices> offsets_;
    Indices counters_;
    Indices limits_;

    RangeProduct(const RangeProduct &t); /* Deactivated */
  };

}; // namespace

#endif // !TENSOR_H
