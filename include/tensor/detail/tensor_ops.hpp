// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_OPS_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_OPS_HPP

namespace tensor {

//
// Unary operations
//
template<typename t>
Tensor<t> operator-(const Tensor<t> &a) {
  Tensor<t> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), std::negate<t>());
  return output;
}

//
// Binary operations
//
//
// TENSOR <OP> TENSOR
//
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const Tensor<t1> &a,
					      const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(), plus<t1,t2>());
  return output;
}

template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const Tensor<t1> &a,
					      const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(), minus<t1,t2>());
  return output;
}

template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const Tensor<t1> &a,
					      const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(), times<t1,t2>());
  return output;
}

template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const Tensor<t1> &a,
					      const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(), divided<t1,t2>());
  return output;
}
//
// TENSOR <OP> NUMBER
//
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const Tensor<t1> &a, const t2 &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), plus_constant<t1,t2>(b));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const Tensor<t1> &a, const t2 &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), minus_constant<t1,t2>(b));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const Tensor<t1> &a, const t2 &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), times_constant<t1,t2>(b));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const Tensor<t1> &a, const t2 &b) {
  Tensor<typename Binop<t1,t2>::type> output(a.dimensions());
  std::transform(a.begin(), a.end(), output.begin(), divided_constant<t1,t2>(b));
  return output;
}
//
// NUMBER <OP> TENSOR
//
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator+(const t1 &a, const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), plus_constant<t2,t1>(a));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator-(const t1 &a, const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), constant_minus<t1,t2>(a));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator*(const t1 &a, const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), times_constant<t2,t1>(a));
  return output;
}
template<typename t1, typename t2>
Tensor<typename Binop<t1,t2>::type> operator/(const t1 &a, const Tensor<t2> &b) {
  Tensor<typename Binop<t1,t2>::type> output(b.dimensions());
  std::transform(b.begin(), b.end(), output.begin(), constant_divided<t1,t2>(a));
  return output;
}


} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_OPS_H
