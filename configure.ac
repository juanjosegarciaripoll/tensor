dnl -*- Autoconf -*-
AC_INIT([C++ Tensor Library],
	[0.1.0],
	[juanjose.garciaripoll@gmail.com],
	[tensor])

#
# Initialize Autoconf. Note that all files in config_aux are created by
# the automatic tools.
#
AC_PREREQ(2.59)
AC_CONFIG_SRCDIR([./README])		dnl Unique file where sources are
AC_CONFIG_MACRO_DIR([m4])		dnl Additional macro directories
AC_CONFIG_AUX_DIR([config_aux])		dnl Where to put autoconf files
AC_CONFIG_HEADERS([config_aux/config.h])	dnl Automatically generated
AC_CONFIG_FILES([Makefile src/Makefile include/Makefile
                 scripts/Makefile test/Makefile www/Makefile
                 essl_lapack/Makefile])
AC_CONFIG_FILES([scripts/tensor-config], [chmod +x scripts/tensor-config])
AC_CONFIG_HEADERS([include/tensor/config.h:include/tensor/config.h.in])

#
# Initialize Automake, minimal version 1.9, with options to make it less
# pedantic, various distribution targets and placing object files together
# inside source dirs.
#
AM_INIT_AUTOMAKE([1.9 foreign dist-bzip2 dist-zip subdir-objects])

#
# Command line arguments
#
#AC_ARG_ENABLE([rpath],
# [AS_HELP_STRING([--enable-rpath],
#     [hard-code the location of the libraries in the binaries])],
# [enable_rpath=yes],
# [enable_rpath=no],
# [enable_rpath=yes])

AC_ARG_WITH([arpack],
 [AS_HELP_STRING([--with-arpack],
     [link in with the arpack-ng library])],
 [],
 [with_arpack=yes])

AC_ARG_WITH([backend],
 [AS_HELP_STRING([--with-backend],
     [BLAS/LAPACK backend (values=AUTO|atlas|cblapack|essl|mkl|veclib)])],
 [with_backend=$withval],
 [with_backend=auto])

AC_ARG_WITH([cblapack],
 [AS_HELP_STRING([--with-cblapack],
     [Choose CBLAPACK backend])],
 [with_backend=cblapack],
 [])

AC_ARG_WITH([fftw],
 [AS_HELP_STRING([--with-fftw],
     [link in with the FFTW library])],
 [],
 [with_fftw=yes])

AC_ARG_ENABLE([threadsafe-deathtest],
 [AS_HELP_STRING([--enable-threadsafe-deathtest],
     [slower but safer mode for running GTest death tests])],
 [],
 [enable_threadsafe_deathtest=yes])

# Programs used to build the library
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O # needed to build in subdirs
AC_LANG([C++])
AC_PROG_LIBTOOL

# OpenMP support (needed for gcc + mkl)
AC_OPENMP

# Size of computer words
TENSOR_BITS

# Numerical libraries
# It also changes the fortran compiler to match those libraries (ESSL)
TENSOR_CHOOSE_LIB

# Functions needed
AC_CHECK_FUNCS_ONCE([gettimeofday])

# Fortran stuff
AC_PROG_F77([ifort gfortran f77])
AC_F77_LIBRARY_LDFLAGS
AC_F77_WRAPPERS
AM_CONDITIONAL(TENSOR_ARPACK, [test $with_arpack = yes])
have_f2c99=no
if test "$F77" = "f77"; then
   f2c_version=`$F77 -v`
   case $f2c_version in
     F2C*C99*)	lt_prog_compiler_pic_F77="$lt_prog_compiler_pic_CXX"
		lt_prog_compiler_wl_F77="$lt_prog_compiler_wl_CXX"
		lt_prog_compiler_static_F77="$lt_prog_compiler_static_CXX"
		lt_prog_compiler_pic_works_F77="$lt_prog_compiler_pic_works_CXX"
		lt_prog_compiler_static_works_F77="$lt_prog_compiler_static_works_CXX"
		
		have_f2c99=yes
		F2C_CPPFLAGS=`f77 --print-cppflags`
		FLIBS="-lf2c"
		LIBS="-lf2c $LIBS"
		;;
     *)		have_f2c99=no
		F2C_CPPFLAGS=""
		;;
   esac
fi
AM_CONDITIONAL(USE_F2C, [test $have_f2c99 = yes])
if test $have_f2c99 = no -a $with_arpack = yes; then
   AC_CHECK_LIB([arpack], [cgetv0_],
   	        [have_arpack=yes; ARPACK_LIBS="-larpack $LIBS"],
		[have_arpack=no])
   AM_CONDITIONAL(BUILD_ARPACK, [test $have_arpack = no])
fi
AC_SUBST(F2C_CPPFLAGS)

# Initialization code for the fortran library
TENSOR_F77_INIT_CODE

# Backtraces
TENSOR_BACKTRACE

# Unit testing with google
TENSOR_GTEST

# Find FFTW library
TENSOR_FFTW

LIBS="$NUM_LIBS $ARPACK_LIBS $FFTW_LIBS $LIBS"

# Disable crap optimization in latest Ubuntu
TENSOR_LD_NO_AS_NEEDED

# Doxygen support

DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(OFF)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN(tensor, doxygen.cfg, doc)

# Missing: headers and library checks

AC_OUTPUT
