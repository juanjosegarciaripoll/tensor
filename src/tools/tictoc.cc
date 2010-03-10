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

#include <ctime>
#include <tensor/tools.h>

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
