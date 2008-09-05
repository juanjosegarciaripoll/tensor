// -*- mode: c++; fill-column: 80; c-basic-offset: 4 -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

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
