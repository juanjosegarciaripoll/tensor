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

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <direct.h>
#define mkdir(x, mode) _mkdir(x)
#else
#include <sys/stat.h>
#endif
#include <tensor/jobs.h>

namespace sdf {

bool make_directory(const std::string &dirname, int mode) {
  std::cout << dirname << std::endl;
  return !mkdir(dirname.c_str(), mode);
}

}  // namespace sdf
