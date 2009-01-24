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

TESTS += test_tensor_unop
check_PROGRAMS += test_tensor_unop
test_tensor_unop_SOURCES = test_tensor_unop.cc
test_tensor_unop_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_tensor_binop
check_PROGRAMS += test_tensor_binop
test_tensor_binop_SOURCES = test_tensor_binop.cc
test_tensor_binop_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_tensor_binop_error
check_PROGRAMS += test_tensor_binop_error
test_tensor_binop_error_SOURCES = test_tensor_binop_error.cc
test_tensor_binop_error_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_tensor_permute
check_PROGRAMS += test_tensor_permute
test_tensor_permute_SOURCES = test_tensor_permute.cc
test_tensor_permute_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_matrix
check_PROGRAMS += test_matrix
test_matrix_SOURCES = test_matrix.cc
test_matrix_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_linalg_svd
check_PROGRAMS += test_linalg_svd
test_linalg_svd_SOURCES = test_linalg_svd.cc
test_linalg_svd_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

noinst_LIBRARIES = libtestmain.a
libtestmain_a_SOURCES = test_main.cc alloc_informer.cc
