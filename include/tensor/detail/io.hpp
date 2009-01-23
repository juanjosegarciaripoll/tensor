// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_IO_H) || defined(TENSOR_DETAIL_IO_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_IO_HPP

namespace tensor {

template<class ForwardIterator>
void write_to_stream(std::ostream &s, ForwardIterator begin,
		     ForwardIterator end) {
  bool first = true;
  for (bool first = true; begin != end; ++begin, first = false) {
    if (!first) s << ", ";
    s << *begin;
  }
}

template<typename elt_t>
std::ostream &operator<<(std::ostream &s, const Vector<elt_t> &t) {
  s << "[";
  write_to_stream(s, t.begin_const(), t.end_const());
  s << "]";
  return s;
}

template<typename elt_t>
std::ostream &operator<<(std::ostream &s, const Tensor<elt_t> &t) {
  s << "(" << t.dimensions() << ")/[";
  write_to_stream(s, t.begin_const(), t.end_const());
  s << "]";
  return s;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_IO_HPP
