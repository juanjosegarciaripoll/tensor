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

#include <cmath>
#include <tensor/flags.h>

namespace tensor {

  const double Flags::DEFAULT = NAN;
  const Flags DEFAULT_FLAGS;
  int Flags::last_key = -1;

  Flags::Flags() :
    _values(0)
  {
  }

  double Flags::get(unsigned int code) const
  {
    if (code >= _values.size())
      return DEFAULT;
    else
      return _values[code];  
  }

  double Flags::get(unsigned int code, double def) const
  {
    double value = get(code);
    if (value == DEFAULT)
      return def;
    return value;
  }

  class Flags &Flags::set(unsigned int code, double value)
  {
    if (code >= _values.size())
      _values.resize(code+1, DEFAULT);
    _values.at(code) = value;
    return *this;
  }

  unsigned int Flags::create_key()
  {
    return last_key++;
  }

}
