dnl -*- Autoconf -*-
AC_INIT([C++ Tensor Library],
	[0.0.0],
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

AC_ARG_ENABLE([mps],
 [AS_HELP_STRING([--enable-mps],
     [Build the MPS library])],
 [enable_mps=yes],
 [enable_mps=no],
 [enable_mps=yes])
AM_CONDITIONAL([BUILD_MPS], [test "x${enable_mps}" != no])

AC_ARG_WITH([arpack],
 [AS_HELP_STRING([--with-arpack],
     [link in with the arpack-ng library])],
 [],
 [with_arpack=yes])

# Programs used to build the library
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])
AC_PROG_LIBTOOL

# Fortran stuff
AC_PROG_F77([f77])
AC_F77_LIBRARY_LDFLAGS
AC_F77_WRAPPERS
AM_CONDITIONAL(TENSOR_ARPACK, [test $with_arpack = yes])
echo FOOLISH $F77
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
		;;
     *)		have_f2c99=no
		F2C_CPPFLAGS=""
		;;
   esac
fi
AM_CONDITIONAL(USE_F2C, [test $have_f2c99=yes])
AC_SUBST(F2C_CPPFLAGS)

# Initialization code for the fortran library
TENSOR_F77_INIT_CODE

# Size of computer words
TENSOR_BITS

# Numerical libraries
TENSOR_CHOOSE_LIB

# Backtraces
TENSOR_BACKTRACE

# Unit testing with google
TENSOR_GTEST

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
