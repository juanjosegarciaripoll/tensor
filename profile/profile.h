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

#ifndef TENSOR_PROFILE_PROFILE_H
#define TENSOR_PROFILE_PROFILE_H

#include <iostream>
#include <tensor/tools.h>
#include "operators.h"

namespace profile {

#define PROF_BEGIN_GROUP(name) \
  std::cout << "<group name='" << name << "'>\n";

#define PROF_BEGIN_SET(name) \
  std::cout << "  <set name='" << name << "'>\n";

#define PROF_ENTRY(set, stmt, repeats)		\
  { double time;				\
  tic();					\
  for (int i = repeats; i; --i) {		\
  stmt;						\
  }						\
  time = toc() / repeats;			\
  std::cout << "   <entry id='" << set << "' time='" << time << "'/>\n"; \
  }

#define PROF_END_SET \
  std::cout << "  </set>\n";


#define PROF_END_GROUP \
  std::cout << "</group>\n";

}

#endif // TENSOR_PROFILE_PROFILE_H
