/* ../arpack-ng/second_NONE.f -- translated by f2c (version 0.1).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include <stdlib.h> /* For exit() */
#include <f2c.h>

/* Subroutine */ int arscnd_(real *t)
{


/*  -- LAPACK auxiliary routine (preliminary version) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     July 26, 1991 */

/*  Purpose */
/*  ======= */

/*  SECOND returns the user time for a process in arscnds. */
/*  This version gets the time from the system function ETIME. */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*      T1 = ETIME( TARRAY ) */
/*      T  = TARRAY( 1 ) */
    *t = 0.f;
    return 0;

/*     End of ARSCND */

} /* arscnd_ */

