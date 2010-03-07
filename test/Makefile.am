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

TESTS += test_index
check_PROGRAMS += test_index
test_index_SOURCES = test_index.cc
test_index_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

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

TESTS += test_view_read
check_PROGRAMS += test_view_read
test_view_read_SOURCES = test_view_read.cc
test_view_read_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_view_set
check_PROGRAMS += test_view_set
test_view_set_SOURCES = test_view_set.cc
test_view_set_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_view_errors
check_PROGRAMS += test_view_errors
test_view_errors_SOURCES = test_view_errors.cc
test_view_errors_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_matrix
check_PROGRAMS += test_matrix
test_matrix_SOURCES = test_matrix.cc
test_matrix_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_mmult
check_PROGRAMS += test_mmult
test_mmult_SOURCES = test_mmult.cc
test_mmult_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_fold
check_PROGRAMS += test_fold
test_fold_SOURCES = test_fold.cc
test_fold_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_linalg_svd
check_PROGRAMS += test_linalg_svd
test_linalg_svd_SOURCES = test_linalg_svd.cc
test_linalg_svd_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

TESTS += test_linalg_eig
check_PROGRAMS += test_linalg_eig
test_linalg_eig_SOURCES = test_linalg_eig.cc
test_linalg_eig_LDADD = libtestmain.a ../src/libtensor.la @GTEST_LDFLAGS@ -lstdc++

noinst_LIBRARIES = libtestmain.a
libtestmain_a_SOURCES = test_main.cc alloc_informer.cc \
	 random_unitary_d.cc random_unitary_z.cc

report.xml: $(check_PROGRAMS)
	echo '<?xml version="1.0" encoding="UTF-8"?>' > $@
	echo '<testframe>' >> $@
	echo '<config field="hostname" value="'`hostname`'"/>' >> $@
	echo '<config field="date" value="'`date +"%Y-%m-%d-%H:%M"`'"/>' >> $@
	for i in CXX CXXFLAGS LDFLAGS; do \
	 k=`sed -e '/S\[.'$${i}'.\]/!d;s/.*="\(.*\)"$$/\1/' ../config.status`; \
	 echo "<config field='$$i' value='$${k}'/>" >> $@ ; \
	 echo "$$i <config field='$$i' value='$${k}'/>" ; \
	done
	for i in $(check_PROGRAMS); do \
	 xmlfile=$${i}.xml; \
	 test -f $$xmlfile || GTEST_OUTPUT=xml:$${xmlfile} ./$$i; \
	 if test -f $$xmlfile; then \
	  sed -e '/<\?xml/d;s/AllTests/'$$i'/g' $${xmlfile} >> $@; \
	 else \
	  echo '<globalfailure name="'$$i'"/>' >> $@; \
	 fi \
	done
	echo '</testframe>' >> $@

report.html: report.xml $(top_srcdir)/scripts/report.xsl
	xsltproc $(top_srcdir)/scripts/report.xsl report.xml > $@

report_nfailures: report.xml
	k=0; \
	for i in test*.xml; do \
	  n=`sed -e '/AllTests/!d;s/.*failures="\(.[^"]*\)".*$/\1/g' $i`; \
	  k=$(($k + $n)); \
	done; \
	echo $k > $@

upload_tests: report.xml report.html
	if [ "x$(NAME)" != "x" ]; then \
	  mkdir $(NAME) && \
	  cp report.xml test*.xml $(NAME) && \
	  rsync -rauvz $(NAME) master:src/tensor_logs/$(NAME); \
	fi
