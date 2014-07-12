// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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

#include <tensor/tensor.h>
#include <functional>
#include "profile.h"

using namespace tensor;
using namespace profile;

template<class binop>
void prof_binop(const char *name, binop &f,
		const int repeats=2*1024,
		const int maxsize=0x10000)
{
  typedef typename binop::first_argument_type A;
  typedef typename binop::second_argument_type B;
  PROF_BEGIN_SET(name) {
    A *a = new A();
    B *b = new B();
    for (int size = 2; size < maxsize; size <<= 2) {
      *a = A::random(maxsize);
      *b = B::random(maxsize);
      PROF_ENTRY(size, f(*a,*b), repeats);
    }
    delete a;
    delete b;
  } PROF_END_SET;
}

template<class binop>
void prof_unop(const char *name, binop &f, const int repeats=2*1024,
	       const int maxsize=0x10000)
{
  typedef typename binop::argument_type A;
  PROF_BEGIN_SET(name) {
    A *a = new A();
    for (int size = 2; size < maxsize; size <<= 2) {
      *a = A::random(maxsize);
      PROF_ENTRY(size, f(*a), repeats);
    }
    delete a;
  } PROF_END_SET;
}

int main()
{

  //
  // VECTOR - VECTOR OPERATIONS
  //

  PROF_BEGIN_GROUP("RTensor") {
    prof_binop("plus", std::plus<RTensor>());
    prof_binop("minus", std::minus<RTensor>());
    prof_binop("multiplies", std::multiplies<RTensor>());
    prof_binop("divides", std::divides<RTensor>());
  } PROF_END_GROUP;


  PROF_BEGIN_GROUP("CTensor") {
    prof_binop("plus", std::plus<CTensor>());
    prof_binop("minus", std::minus<CTensor>());
    prof_binop("multiplies", std::multiplies<CTensor>());
    prof_binop("divides", std::divides<CTensor>());
  } PROF_END_GROUP;

  //
  // VECTOR - NUMBER OPERATIONS
  //

  PROF_BEGIN_GROUP("RTensor") {
    double value = 3;
    prof_unop("plus3", plusN<RTensor,double>(value));
    prof_unop("minus3", minusN<RTensor,double>(value));
    prof_unop("multiplies3", multipliesN<RTensor,double>(value));
    prof_unop("divides3", dividesN<RTensor,double>(value));
  } PROF_END_GROUP;


  PROF_BEGIN_GROUP("CTensor") {
    cdouble value = to_complex(0,3);
    prof_unop("plus3", plusN<CTensor,cdouble>(value));
    prof_unop("minus3", minusN<CTensor,cdouble>(value));
    prof_unop("multiplies3", multipliesN<CTensor,cdouble>(value));
    prof_unop("divides3", dividesN<CTensor,cdouble>(value));
  } PROF_END_GROUP;

}
