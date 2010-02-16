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
AC_CONFIG_FILES([Makefile src/Makefile scripts/Makefile test/Makefile])
AC_CONFIG_FILES([scripts/tensor-config], [chmod +x scripts/tensor-config])
AC_CONFIG_HEADERS([include/tensor/config.h:include/tensor/config.h.in])

#
# Initialize Automake, minimal version 1.9, with options to make it less
# pedantic, various distribution targets and placing object files together
# inside source dirs.
#
AM_INIT_AUTOMAKE([1.9 foreign dist-bzip2 dist-zip subdir-objects])

# Programs used to build the library
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])
AC_PROG_LIBTOOL

# Numerical libraries
TENSOR_VECLIB
TENSOR_ATLAS

# Unit testing with google
if which gtest-config >/dev/null; then
  HAVE_GTEST=1
  GTEST_CFLAGS=`gtest-config --cppflags --cxxflags`
  GTEST_LDFLAGS=`gtest-config --ldflags --libs`
fi
AM_CONDITIONAL([HAVE_GTEST], [test "x$HAVE_GTEST" != x])
AC_SUBST(GTEST_CFLAGS)
AC_SUBST(GTEST_LDFLAGS)

# Size of computer words
AM_CONDITIONAL([HAVE_64BITS], [false])

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
