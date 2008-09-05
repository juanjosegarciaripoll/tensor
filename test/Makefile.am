# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include @GTEST_CFLAGS@

#
# No rules here yet
#
TESTS =
check_PROGRAMS =

TESTS += test_refcount
check_PROGRAMS += test_refcount
test_refcount_SOURCES = test_refcount.cc test_main.cc
test_refcount_LDADD = ../src/libtensor.la @GTEST_LDFLAGS@
