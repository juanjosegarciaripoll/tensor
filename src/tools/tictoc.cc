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

#include <tensor/config.h>
#include <ctime>
#ifdef HAVE_GETTIMEOFDAY
#include <sys/time.h>
#endif
#include <tensor/tools.h>

namespace tensor {

/* TODO: Use C++ clock routines as in profile.cc */

static double now() {
#if defined(HAVE_GETTIMEOFDAY)
  struct timeval tic_now;
  gettimeofday(&tic_now, nullptr);
  double seconds = tic_now.tv_sec;
  double museconds = tic_now.tv_usec;
  return seconds + 1e-6 * museconds;
#else
  return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
#endif
}

static double sometime;

/**Reset the time counter.*/
double tic() { return sometime = now(); }

/**Output the time in seconds since last invocation of tic(). This function
     only counts the real time that the program has used since the last
     call of tic(). This may not be related to the processing time if your
     program is using more than one core and thus it may not be very accurate
     for computing CPU consumption in clusters.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
double toc() { return now() - sometime; }

/**Output the time in seconds since the given time. This function
     only counts the real time that the program has used since the last
     call of tic(). This may not be related to the processing time if your
     program is using more than one core and thus it may not be very accurate
     for computing CPU consumption in clusters.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
double toc(double when) { return now() - when; }

}  // namespace tensor
