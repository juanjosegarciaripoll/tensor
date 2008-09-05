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
AC_CONFIG_AUX_DIR([config_aux])		dnl Where to put autoconf files
AC_CONFIG_HEADERS([config_aux/config.h])	dnl Automatically generated
AC_CONFIG_FILES([Makefile src/Makefile scripts/Makefile test/Makefile])
AC_CONFIG_FILES([scripts/tensor-config], [chmod +x scripts/tensor-config])

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

# Unit testing with google
if which gtest-config >/dev/null; then
  HAVE_GTEST=1
  GTEST_CFLAGS=`gtest-config --cppflags --cxxflags`
  GTEST_LDFLAGS=`gtest-config --ldflags --libs`
fi
AM_CONDITIONAL([HAVE_GTEST], [test "x$HAVE_GTEST" != x])
AC_SUBST(GTEST_CFLAGS)
AC_SUBST(GTEST_LDFLAGS)

# Missing: headers and library checks

AC_OUTPUT
