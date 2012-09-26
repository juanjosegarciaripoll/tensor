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
AC_ARG_WITH([mkl-dir],
 [AS_HELP_STRING([--with-mkl-dir],
     [provide location of the Intel MKL Library])],
 [MKL_DIR=$with_mkl_dir],
 [])

# Programs used to build the library
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])
AC_PROG_LIBTOOL

# Size of computer words
TENSOR_BITS

# Numerical libraries
TENSOR_CHOOSE_LIB

# Unit testing with google
if test -d "${top_srcdir}/test/googletest-read-only"; then
  GTEST_DIR="${top_srcdir}/test/googletest-read-only"
else
  if test -d "${top_srcdir}/test/gtest-1.6.0"; then
    GTEST_DIR="${top_srcdir}/test/gtest-1.6.0"
  else
    test -d "`pwd`/test" || mkdir "`pwd`/test"
    ls -l test
    if (which unzip > /dev/null && which curl > /dev/null && \
        cd test && \
        curl "http://googletest.googlecode.com/files/gtest-1.6.0.zip" > tmp.zip && \
        unzip -x tmp.zip -d "`pwd`" && \
        rm tmp.zip); then
      GTEST_DIR="`pwd`/test/gtest-1.6.0"
    else
      GTEST_DIR=""
      AC_MSG_WARN([Please download google test library in `pwd`/test/])
    fi
    ls -l test
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
