# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_builddir)/include

#
# Main library
#
lib_LTLIBRARIES = libtensor.la

libtensor_la_SOURCES = \
	tools/tictoc.cc \
	rand/rand.cc \
	indices/indices.cc \
	sparse/full_d.cc \
	sparse/full_z.cc \
	sparse/sparse_d.cc \
	sparse/sparse_z.cc \
	sparse/sparse_d_to_z.cc \
	sparse/sparse_kron_d.cc \
	sparse/sparse_kron_z.cc \
	sparse/sparse_ops_d.cc \
	sparse/sparse_ops_z.cc \
	tensor/tensor_common.cc \
	tensor/tensor_d.cc \
	tensor/tensor_z.cc \
	tensor/tensor_to_complex.cc \
	tensor/tensor_conj.cc \
	tensor/matrix_norminf_d.cc \
	tensor/matrix_norminf_z.cc \
	tensor/matrix_adjoint_z.cc \
	tensor/matrix_diag_d.cc \
	tensor/matrix_diag_z.cc \
	tensor/matrix_trace_d.cc \
	tensor/matrix_trace_z.cc \
	tensor/matrix_transpose_d.cc \
	tensor/matrix_transpose_z.cc \
	tensor/tensor_permute_d.cc \
	tensor/tensor_permute_z.cc \
	tensor/tensor_fold_d.cc \
	tensor/tensor_fold_z.cc \
	tensor/tensor_fold_dz.cc \
	tensor/tensor_foldin_d.cc \
	tensor/tensor_foldin_z.cc \
	tensor/tensor_kron_d.cc \
	tensor/tensor_kron_z.cc \
	tensor/tensor_sum_d.cc \
	tensor/tensor_sum_z.cc \
	tensor/tensor_norm0_d.cc \
	tensor/tensor_norm0_z.cc \
	tensor/tensor_norm2_d.cc \
	tensor/tensor_norm2_z.cc \
	tensor/tensor_scale_d.cc \
	tensor/tensor_scale_z.cc \
	generated/tensor_unop_abs_d.cc \
	generated/tensor_unop_sqrt_d.cc \
	generated/tensor_unop_exp_d.cc \
	generated/tensor_unop_cos_d.cc \
	generated/tensor_unop_sin_d.cc \
	generated/tensor_unop_tan_d.cc \
	generated/tensor_unop_cosh_d.cc \
	generated/tensor_unop_sinh_d.cc \
	generated/tensor_unop_tanh_d.cc \
	generated/tensor_unop_abs_z.cc \
	generated/tensor_unop_cos_z.cc \
	generated/tensor_unop_sin_z.cc \
	generated/tensor_unop_tan_z.cc \
	generated/tensor_unop_cosh_z.cc \
	generated/tensor_unop_sinh_z.cc \
	generated/tensor_unop_tanh_z.cc \
	generated/tensor_unop_exp_z.cc \
	generated/tensor_unop_sqrt_z.cc \
	linalg/eig_d.cc \
	linalg/eig_z.cc \
	linalg/svd_d.cc \
	linalg/svd_z.cc \
	linalg/solve_d.cc \
	linalg/solve_z.cc \
	linalg/expm_d.cc \
	linalg/expm_z.cc \
	linalg/eig_power_d.cc \
	linalg/eig_power_z.cc \
	views/range.cc

lib_LTLIBRARIES += libmps.la

libmps_la_SOURCES = \
	mps/quantum/spin.cc

generated/tensor_unop_%_d.cc: tensor/tensor_unop.cc Makefile
	k=`echo $@ | sed 's,.*unop_\(.*\)_d.cc,\1,g'`;\
	sed -e "s,TYPE[12],Tensor<double>,g;s,OPERATOR1,$$k,;s,OPERATOR2,std::$$k,g" $< > $@
generated/tensor_unop_%_z.cc: tensor/tensor_unop.cc Makefile
	k=`echo $@ | sed 's,.*unop_\(.*\)_z.cc,\1,g'`;\
	sed -e "s,TYPE[12],Tensor<cdouble>,g;s,OPERATOR1,$$k,;s,OPERATOR2,std::$$k,g" $< > $@
generated/tensor_unop_abs_d.cc: tensor/tensor_unop.cc Makefile
	sed -e "s,TYPE1,Tensor<double>,g;s,TYPE2,Tensor<double>,g;s,OPERATOR1,abs,;s,OPERATOR2,std::abs,g" $< > $@
generated/tensor_unop_abs_z.cc: tensor/tensor_unop.cc Makefile
	sed -e "s,TYPE1,Tensor<cdouble>,g;s,TYPE2,Tensor<double>,g;s,OPERATOR1,abs,;s,OPERATOR2,std::abs,g" $< > $@

if HAVE_64BITS
libtensor_la_SOURCES += rand/rand64.cc
else
libtensor_la_SOURCES += rand/rand32.cc
endif
