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

#include <tensor/config.h>
#include <ctime>
#ifdef HAVE_GETTIMEOFDAY
#include <sys/time.h>
#endif
#include <tensor/tools.h>

namespace tensor {

#if defined(HAVE_GETTIMEOFDAY)

  struct timeval tic_start;

  /**Reset the time counter.*/
  void tic()
  {
    gettimeofday(&tic_start, NULL);
  }

  /**Output the time in seconds since last invocation of tic(). This function
     only counts the real time that the program has used since the last
     call of tic(). This may not be related to the processing time if your
     program is using more than one core and thus it may not be very accurate
     for computing CPU consumption in clusters.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
  double toc()
  {
    struct timeval tic_now;
    gettimeofday(&tic_now, NULL);
    double seconds = tic_now.tv_sec - tic_start.tv_sec;
    double museconds = tic_now.tv_usec - tic_start.tv_usec;
    return seconds + 1e-6 * museconds;
  }

#endif

#if !defined(HAVE_GETTIMEOFDAY)

  static clock_t tic_start;

  /**Reset the time counter.*/
  void tic()
  {
    tic_start = clock();
  }

  /**Output the time in seconds since last invocation of tic(). This function
     only counts the processing time that the program has used since the last
     call of tic(). If there are more than one program running in the computer,
     or if your program makes use of functions such as sleep() to make pauses,
     this time will be much shorter than the real elapsed time.

     Opposite to Matlab, toc() by itself does not produce any informative message.
  */
  double toc()
  {
    return (clock()-tic_start)/((double)CLOCKS_PER_SEC);
  }

#endif

}
