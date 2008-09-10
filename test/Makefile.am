# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/src @GTEST_CFLAGS@

#
# No rules here yet
#
TESTS =
check_PROGRAMS =

TESTS += test_refcount
check_PROGRAMS += test_refcount
test_refcount_SOURCES = test_refcount.cc
test_refcount_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_rand
check_PROGRAMS += test_rand
test_rand_SOURCES = test_rand.cc
test_rand_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_tensor
check_PROGRAMS += test_tensor
test_tensor_SOURCES = test_tensor_base.cc \
	test_tensor_getset.cc
test_tensor_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_matrix
check_PROGRAMS += test_matrix
test_matrix_SOURCES = test_matrix.cc
test_matrix_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

noinst_LIBRARIES = libtestmain.a
libtestmain_a_SOURCES = test_main.cc alloc_informer.cc
