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

#ifndef TENSOR_OPERATORS_OPERATORS_H
#define TENSOR_OPERATORS_OPERATORS_H

#include <iostream>
#include <tensor/tools.h>

namespace profile {

  template<class A, class B> struct plusN {
    typedef A argument_type;
    const B value;
    plusN(const B n) : value (n) {}
    A operator()(const A &v) { return v + value; }
  };

  template<class A, class B> struct minusN {
    typedef A argument_type;
    const B value;
    minusN(const B n) : value (n) {}
    A operator()(const A &v) { return v - value; }
  };

  template<class A, class B> struct multipliesN {
    typedef A argument_type;
    const B value;
    multipliesN(const B n) : value (n) {}
    A operator()(const A &v) { return v * value; }
  };

  template<class A, class B> struct dividesN {
    typedef A argument_type;
    const B value;
    dividesN(const B n) : value (n) {}
    A operator()(const A &v) { return v / value; }
  };

}

#endif // TENSOR_OPERATORS_OPERATORS_H
