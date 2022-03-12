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

#include <chrono>
#include <ratio>
#include <tensor/config.h>
#include <tensor/tools.h>

namespace tensor {

static std::chrono::steady_clock inner_clock;
static auto now = inner_clock.now();

/**Reset the time counter.*/
double tic() {
  now = inner_clock.now();
  return 0.0;
}

/**Output the time in seconds since last invocation of tic(). This function
     only counts the real time that the program has used since the last
     call of tic(). This may not be related to the processing time if your
     program is using more than one core and thus it may not be very accurate
     for computing CPU consumption in clusters.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
double toc() {
  auto duration = inner_clock.now() - now;
  return std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(
             duration)
             .count() *
         1e-9;
}

/**Output the time in seconds since the given time. This function
     only counts the real time that the program has used since the last
     call of tic(). This may not be related to the processing time if your
     program is using more than one core and thus it may not be very accurate
     for computing CPU consumption in clusters.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
double toc(double when) { return toc() - when; }

}  // namespace tensor
