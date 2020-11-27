// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2013 Juan Jose Garcia Ripoll

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

#include <stdio.h>
#include <tensor/sdf.h>

namespace sdf {

bool rename_file(const std::string &orig, const std::string &dest,
                 bool overwrite) {
  if (file_exists(dest)) {
    if (!overwrite || !delete_file(dest)) {
      std::cerr << "Unable to move file to destination " << dest
                << " because destination cannot be deleted." << std::endl;
      abort();
    }
  }
  if (!file_exists(orig)) {
    std::cerr << "In rename_file(), original file " << dest << " does not exist"
              << std::endl;
    abort();
  }
  return rename(orig.c_str(), dest.c_str()) == 0;
}

}  // namespace sdf
