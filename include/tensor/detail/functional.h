// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_DETAIL_FUNCTIONAL_H
#define TENSOR_DETAIL_FUNCTIONAL_H

namespace tensor {

template<typename t1, typename t2> struct Binop;

template<typename t>
struct Binop<t,t> {
  typedef t type;
  type operator()(const t &, const t &);
};
  
template<typename t>
struct Binop<t,std::complex<t> > {
  typedef std::complex<t> type;
  type operator()(const t &, const std::complex<t> &);
};
  
template<typename t>
struct Binop<std::complex<t>,t> {
  typedef std::complex<t> type;
  type operator()(const std::complex<t> &, const t &);
};

//
// Binary operations based on previous types
//

template<typename t1, typename t2>
class plus {
  typename Binop<t1,t2>::type operator()(const t1 &a, const t2 &b) {
    return a + b;
  }
};

template<typename t1, typename t2>
class minus {
  typename Binop<t1,t2>::type operator()(const t1 &a, const t2 &b) {
    return a + b;
  }
};

template<typename t1, typename t2>
class times {
  typename Binop<t1,t2>::type operator()(const t1 &a, const t2 &b) {
    return a + b;
  }
};

template<typename t1, typename t2>
class divided {
  typename Binop<t1,t2>::type operator()(const t1 &a, const t2 &b) {
    return a + b;
  }
};

//
// Unary operations against constants
//
template<typename t1, typename t2>
class plus_constant {
  const t2 &value;
  plus_constant(const t2 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t1 &a) {
    return a + value;
  }
};

template<typename t1, typename t2>
class minus_constant {
  const t2 &value;
  minus_constant(const t2 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t1 &a) {
    return a - value;
  }
};

template<typename t1, typename t2>
class constant_minus {
  const t1 &value;
  constant_minus(const t1 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t2 &a) {
    return value - a;
  }
};

template<typename t1, typename t2>
class times_constant {
  const t2 &value;
  times_constant(const t2 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t1 &a) {
    return a * value;
  }
};

template<typename t1, typename t2>
class divided_constant {
  const t2 &value;
  divided_constant(const t2 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t1 &a) {
    return a / value;
  }
};

template<typename t1, typename t2>
class constant_divided {
  const t1 &value;
  constant_divided(const t1 &b) : value(b) {}
  typename Binop<t1,t2>::type operator()(const t2 &a) {
    return value / a;
  }
};




} // namespace tensor

#endif // !TENSOR_DETAIL_FUNCTIONAL_H
