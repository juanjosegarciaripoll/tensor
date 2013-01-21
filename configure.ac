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
                 essl_lapack/Makefile arpack-ng/Makefile])
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
AC_ARG_ENABLE([rpath],
 [AS_HELP_STRING([--enable-rpath],
     [hard-code the location of the libraries in the binaries])],
 [enable_rpath=yes],
 [enable_rpath=no],
 [enable_rpath=yes])

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

# Size of computer words
TENSOR_BITS

# Numerical libraries
TENSOR_CHOOSE_LIB

# Backtraces
TENSOR_BACKTRACE

# Unit testing with google
AC_MSG_CHECKING([for googletest-read-only in sources])
if test -d "${ac_confdir}/test/googletest-read-only"; then
  AC_MSG_RESULT([yes])
  GTEST_DIR="${ac_confdir}/test/googletest-read-only"
else
  AC_MSG_RESULT([no])
  AC_MSG_CHECKING([for gtest-1.6.0 in sources])	
  if test -d "${ac_confdir}/test/gtest-1.6.0"; then
    AC_MSG_RESULT([yes])
    GTEST_DIR="${ac_confdir}/test/gtest-1.6.0"
  else
    AC_MSG_RESULT([no])
    test -d "`pwd`/test" || mkdir "`pwd`/test"
    AC_MSG_CHECKING([trying to download Google Test library])
    if (which unzip && which curl && \
        cd test && \
        curl "http://googletest.googlecode.com/files/gtest-1.6.0.zip" > tmp.zip && \
        unzip -x tmp.zip -d "`pwd`" && \
        rm tmp.zip) >&AS_MESSAGE_LOG_FD; then
      AC_MSG_RESULT([worked])
      GTEST_DIR="`pwd`/test/gtest-1.6.0"
    else
      AC_MSG_RESULT([unable])
      GTEST_DIR=""
      AC_MSG_WARN([Please download google test library in `pwd`/test/])
    fi
  fi
fi
AM_CONDITIONAL([HAVE_GTEST], [test "x${GTEST_DIR}" != x])
AC_SUBST(GTEST_DIR)

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
