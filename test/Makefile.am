# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/src \
	-I$(top_builddir)/include -I"@GTEST_DIR@/include" -I"@GTEST_DIR@" 

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_LDFLAGS = libgtest_main.a $(LDFLAGS) -lpthread

#
# No rules here yet
#
TESTS =
check_PROGRAMS =

AM_TESTS_FD_REDIRECT = --gtest_death_test_style=threadsafe

#####################################################################
# TENSOR LIBRARY TESTS
#

TESTS += test_refcount
check_PROGRAMS += test_refcount
test_refcount_SOURCES = test_refcount.cc
test_refcount_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_index
check_PROGRAMS += test_index
test_index_SOURCES = test_index.cc
test_index_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_rand
check_PROGRAMS += test_rand
test_rand_SOURCES = test_rand.cc
test_rand_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor
check_PROGRAMS += test_tensor
test_tensor_SOURCES = test_tensor_base.cc \
	test_tensor_getset.cc
test_tensor_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_unop
check_PROGRAMS += test_tensor_unop
test_tensor_unop_SOURCES = test_tensor_unop.cc
test_tensor_unop_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_binop
check_PROGRAMS += test_tensor_binop
test_tensor_binop_SOURCES = test_tensor_binop.cc
test_tensor_binop_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_binop_error
check_PROGRAMS += test_tensor_binop_error
test_tensor_binop_error_SOURCES = test_tensor_binop_error.cc
test_tensor_binop_error_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_kron
check_PROGRAMS += test_tensor_kron
test_tensor_kron_SOURCES = test_tensor_kron.cc
test_tensor_kron_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_permute
check_PROGRAMS += test_tensor_permute
test_tensor_permute_SOURCES = test_tensor_permute.cc
test_tensor_permute_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_view_read
check_PROGRAMS += test_view_read
test_view_read_SOURCES = test_view_read.cc
test_view_read_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_view_set
check_PROGRAMS += test_view_set
test_view_set_SOURCES = test_view_set.cc
test_view_set_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_view_errors
check_PROGRAMS += test_view_errors
test_view_errors_SOURCES = test_view_errors.cc
test_view_errors_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_matrix
check_PROGRAMS += test_matrix
test_matrix_SOURCES = test_matrix.cc
test_matrix_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_sparse_base
check_PROGRAMS += test_sparse_base
test_sparse_base_SOURCES = test_sparse_base.cc
test_sparse_base_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_sparse_kron
check_PROGRAMS += test_sparse_kron
test_sparse_kron_SOURCES = test_sparse_kron.cc
test_sparse_kron_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_sparse_indices
check_PROGRAMS += test_sparse_indices
test_sparse_indices_SOURCES = test_sparse_indices.cc
test_sparse_indices_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mmult
check_PROGRAMS += test_mmult
test_mmult_SOURCES = test_mmult.cc
test_mmult_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_fold
check_PROGRAMS += test_fold
test_fold_SOURCES = test_fold.cc
test_fold_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_svd
check_PROGRAMS += test_linalg_svd
test_linalg_svd_SOURCES = test_linalg_svd.cc
test_linalg_svd_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_eig
check_PROGRAMS += test_linalg_eig
test_linalg_eig_SOURCES = test_linalg_eig.cc
test_linalg_eig_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_eig_sym
check_PROGRAMS += test_linalg_eig_sym
test_linalg_eig_sym_SOURCES = test_linalg_eig_sym.cc
test_linalg_eig_sym_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_eig_power
check_PROGRAMS += test_linalg_eig_power
test_linalg_eig_power_SOURCES = test_linalg_eig_power.cc
test_linalg_eig_power_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_expm
check_PROGRAMS += test_linalg_expm
test_linalg_expm_SOURCES = test_linalg_expm.cc
test_linalg_expm_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_jobs
check_PROGRAMS += test_jobs
test_jobs_SOURCES = test_jobs.cc
test_jobs_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_sdf
check_PROGRAMS += test_sdf
test_sdf_SOURCES = test_sdf.cc
test_sdf_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_tensor_take_diag
check_PROGRAMS += test_tensor_take_diag
test_tensor_take_diag_SOURCES = test_tensor_take_diag.cc
test_tensor_take_diag_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_linalg_eigs
check_PROGRAMS += test_linalg_eigs
test_linalg_eigs_SOURCES = test_linalg_eigs.cc
test_linalg_eigs_LDADD = libtestmain.a ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

#####################################################################
# MPS LIBRARY TESTS
#

TESTS += test_quantum_spin
check_PROGRAMS += test_quantum_spin
test_quantum_spin_SOURCES = test_quantum_spin.cc
test_quantum_spin_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_quantum_fock
check_PROGRAMS += test_quantum_fock
test_quantum_fock_SOURCES = test_quantum_fock.cc
test_quantum_fock_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_base
check_PROGRAMS += test_itebd_base
test_itebd_base_SOURCES = test_itebd_base.cc
test_itebd_base_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_expected
check_PROGRAMS += test_itebd_expected
test_itebd_expected_SOURCES = test_itebd_expected.cc
test_itebd_expected_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_itime
check_PROGRAMS += test_itebd_itime
test_itebd_itime_SOURCES = test_itebd_itime.cc
test_itebd_itime_LDADD = libtestmain.a ../src/libmps.la ../src/libtensor.la $(GTEST_LDFLAGS) #-lstdc++

noinst_LIBRARIES = libtestmain.a
libtestmain_a_SOURCES = test_main.cc alloc_informer.cc \
	 random_unitary_d.cc random_unitary_z.cc random_permutation.cc

#
# Build the Google Test library. The way this library has changed: we now have to
# compile it for every project. However, since we have downloaded the library, we
# cannot rely on automake and we have to generate the sources.
#

noinst_LIBRARIES += libgtest.a
libgtest_a_SOURCES = gtest-all.cc

noinst_LIBRARIES += libgtest_main.a
libgtest_main_a_SOURCES = gtest-all.cc gtest_main.cc

gtest-all.cc gtest_main.cc :
	cp @GTEST_DIR@/src/$@ .

OUTPUT_FLAGS = CXX CXXFLAGS LDFLAGS \
	TENSOR_USE_VECLIB TENSOR_USE_MKL TENSOR_USE_ATLAS \
	host_cpu host_os host_vendor

report.xml: $(check_PROGRAMS) Makefile
	echo '<?xml version="1.0" encoding="UTF-8"?>' > $@
	echo '<testframe>' >> $@
	echo '<config field="hostname" value="'`hostname`'"/>' >> $@
	echo '<config field="date" value="'`date +"%Y-%m-%d-%H:%M"`'"/>' >> $@
	for i in $(OUTPUT_FLAGS); do \
	 k=`sed -e '/[DS]\[.'$${i}'.\]/!d;s/.*="\(.*\)"$$/\1/' ../config.status`; \
	 if [ "x$$k" = "x" ]; then k="0"; fi; \
	 echo "<config field='$$i' value='$${k}'/>" >> $@ ; \
	 echo "$$i <config field='$$i' value='$${k}'/>" ; \
	done
	-shared=`sed '/^enable_shared=/!d;s,enable_shared=\(.*\)$,\1,'`; \
	echo '<config field="SHARED" value="'$$shared'"/>' >> $@
	for i in $(check_PROGRAMS); do \
	 xmlfile=$${i}.xml; \
	 test -f $$xmlfile || GTEST_OUTPUT=xml:$${xmlfile} ./$$i; \
	 test -f $$xmlfile || echo '<globalfailure name="'$$i'"/>' >> $$xmlfile; \
	 sed -e '/<\?xml/d;s/AllTests/'$$i'/g' $${xmlfile} >> $@; \
	done
	echo '</testframe>' >> $@

report.html: report.xml $(top_srcdir)/scripts/report.xsl
	xsltproc $(top_srcdir)/scripts/report.xsl report.xml > $@

report_nfailures: report.xml
	k=0; \
	for i in test*.xml; do \
	  n=`sed -e '/AllTests/!d;s/.*failures="\(.[^"]*\)".*$$/\1/g' $$i`; \
	  if [ "x$$n" = "x" ]; then n=1; fi; k=$$(($$k + $$n)); \
	done; \
	echo $$k > $@

upload_tests: report.xml report_nfailures
	if [ "x$(NAME)" != "x" ]; then \
	  host=$(HOST); \
	  if [ "x$${host}" = "x" ]; then host="master:"; fi; \
	  if [ ! -d $(NAME) ]; then mkdir $(NAME) ; fi;  \
	  cp report.xml report_nfailures test*.xml $(NAME) && \
	  rsync -rauvz $(NAME) $${host}src/tensor_logs/; \
	fi

CLEANFILES = *.xml report_nfailures
