/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#pragma once
#ifndef TENSOR_MAP_H
#define TENSOR_MAP_H

#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace tensor {

template <class Tensor>
class Map {
 public:
  virtual ~Map(){};
  virtual const Tensor operator()(const Tensor &arg) const { return arg; };
};

template <class Matrix>
class MatrixMap : public Map<Tensor<typename Matrix::elt_t> > {
 public:
  typedef Tensor<typename Matrix::elt_t> tensor_t;
  MatrixMap(const Matrix &m, bool transpose = false);
  virtual ~MatrixMap();
  virtual const tensor_t operator()(const tensor_t &arg) const;

 private:
  const Matrix m_;
  const bool transpose_;
};

template <class Func, class Tensor>
class FunctionMap : public Map<Tensor> {
 public:
  FunctionMap(const Func &f) : f_(f) {}
  virtual ~FunctionMap(){};
  virtual const Tensor operator()(const Tensor &arg) const { return f_(arg); }

 private:
  const Func &f_;
};

template <class out, class arg0, class arg1, class par1>
struct Closure1 {
  typedef out (*f_ptr)(arg0, arg1);
  Closure1(f_ptr f, par1 &a1) : f_(f), a1_(a1) {}
  out operator()(arg0 a0) const { return (*f_)(a0, a1_); }

 private:
  const f_ptr f_;
  par1 a1_;
};

template <class out, class arg0, class arg1, class par1>
inline Closure1<out, arg0, arg1, par1> with_args(out (*f)(arg0, arg1),
                                                 par1 a1) {
  return Closure1<out, arg0, arg1, par1>(f, a1);
}

template <class out, class arg0, class arg1, class arg2, class par1, class par2>
struct Closure2 {
  typedef out (*f_ptr)(arg0, arg1, arg2);
  Closure2(f_ptr f, par1 &a1, par2 &a2) : f_(f), a1_(a1), a2_(a2) {}
  out operator()(arg0 a0) const { return (*f_)(a0, a1_, a2_); }

 private:
  const f_ptr f_;
  par1 a1_;
  par2 a2_;
};

template <class out, class arg0, class arg1, class arg2, class par1, class par2>
inline Closure2<out, arg0, arg1, arg2, par1, par2> with_args(
    out (*f)(arg0, arg1, arg2), par1 a1, par2 a2) {
  return Closure2<out, arg0, arg1, arg2, par1, par2>(f, a1, a2);
}

template <class out, class arg0, class arg1, class arg2, class arg3, class par1,
          class par2, class par3>
struct Closure3 {
  typedef out (*f_ptr)(arg0, arg1, arg2, arg3);
  Closure3(f_ptr f, par1 &a1, par2 &a2, par3 &a3)
      : f_(f), a1_(a1), a2_(a2), a3_(a3) {}
  out operator()(arg0 a0) const { return (*f_)(a0, a1_, a2_, a3_); }

 private:
  const f_ptr f_;
  par1 a1_;
  par2 a2_;
  par3 a3_;
};

template <class out, class arg0, class arg1, class arg2, class arg3, class par1,
          class par2, class par3>
inline Closure3<out, arg0, arg1, arg2, arg3, par1, par2, par3> with_args(
    out (*f)(arg0, arg1, arg2, arg3), par1 a1, par2 a2, par3 a3) {
  return Closure3<out, arg0, arg1, arg2, arg3, par1, par2, par3>(f, a1, a2, a3);
}

template <class out, class arg0, class arg1, class arg2, class arg3, class arg4,
          class par1, class par2, class par3, class par4>
struct Closure4 {
  typedef out (*f_ptr)(arg0, arg1, arg2, arg3, arg4);
  Closure4(f_ptr f, par1 &a1, par2 &a2, par3 &a3, par4 &a4)
      : f_(f), a1_(a1), a2_(a2), a3_(a3), a4_(a4) {}
  out operator()(arg0 a0) const { return (*f_)(a0, a1_, a2_, a3_, a4_); }

 private:
  const f_ptr f_;
  par1 a1_;
  par2 a2_;
  par3 a3_;
  par4 a4_;
};

template <class out, class arg0, class arg1, class arg2, class arg3, class arg4,
          class par1, class par2, class par3, class par4>
inline Closure4<out, arg0, arg1, arg2, arg3, arg4, par1, par2, par3, par4>
with_args(out (*f)(arg0, arg1, arg2, arg3, arg4), par1 a1, par2 a2, par3 a3,
          par4 a4) {
  return Closure4<out, arg0, arg1, arg2, arg3, arg4, par1, par2, par3, par4>(
      f, a1, a2, a3, a4);
}

template <class out, class arg0, class arg1, class arg2, class arg3, class arg4,
          class arg5, class par1, class par2, class par3, class par4,
          class par5>
struct Closure5 {
  typedef out (*f_ptr)(arg0, arg1, arg2, arg3, arg4, arg5);
  Closure5(f_ptr f, par1 &a1, par2 &a2, par3 &a3, par4 &a4, par5 &a5)
      : f_(f), a1_(a1), a2_(a2), a3_(a3), a4_(a4), a5_(a5) {}
  out operator()(arg0 a0) const { return (*f_)(a0, a1_, a2_, a3_, a4_, a5_); }

 private:
  const f_ptr f_;
  par1 a1_;
  par2 a2_;
  par3 a3_;
  par4 a4_;
  par5 a5_;
};

template <class out, class arg0, class arg1, class arg2, class arg3, class arg4,
          class arg5, class par1, class par2, class par3, class par4,
          class par5>
inline Closure5<out, arg0, arg1, arg2, arg3, arg4, arg5, par1, par2, par3, par4,
                par5>
with_args(out (*f)(arg0, arg1, arg2, arg3, arg4, arg5), par1 a1, par2 a2,
          par3 a3, par4 a4, par5 a5) {
  return Closure5<out, arg0, arg1, arg2, arg3, arg4, arg5, par1, par2, par3,
                  par4, par5>(f, a1, a2, a3, a4, a5);
}

extern template class MatrixMap<RTensor>;
extern template class MatrixMap<CTensor>;
extern template class MatrixMap<RSparse>;
extern template class MatrixMap<CSparse>;

}  // namespace tensor

#endif  // TENSOR_MAP_H
