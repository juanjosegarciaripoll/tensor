# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include

#
# Main library
#
lib_LTLIBRARIES = libtensor.la

libtensor_la_SOURCES = \
	tools/tictoc.cc \
	rand/rand.cc \
	indices/indices.cc \
	tensor/tensor_common.cc \
	tensor/tensor_d.cc \
	tensor/tensor_z.cc \
	tensor/matrix_diag_d.cc \
	tensor/matrix_diag_z.cc \
	tensor/matrix_trace_d.cc \
	tensor/matrix_trace_z.cc \
	tensor/matrix_transpose_d.cc \
	tensor/matrix_transpose_z.cc \
	tensor/tensor_permute_d.cc \
	tensor/tensor_permute_z.cc \
	generated/tensor_unop_abs_d.cc \
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
	generated/tensor_unop_exp_z.cc

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

pkginclude_HEADERS = \
	$(top_srcdir)/include/tensor/detail/functional.h \
	$(top_srcdir)/include/tensor/detail/io.hpp \
	$(top_srcdir)/include/tensor/detail/tensor_base.hpp \
	$(top_srcdir)/include/tensor/detail/tensor_matrix.hpp \
	$(top_srcdir)/include/tensor/detail/tensor_ops.hpp \
	$(top_srcdir)/include/tensor/detail/tensor_reshape.hpp \
	$(top_srcdir)/include/tensor/indices.h \
	$(top_srcdir)/include/tensor/io.h \
	$(top_srcdir)/include/tensor/numbers.h \
	$(top_srcdir)/include/tensor/rand.h \
	$(top_srcdir)/include/tensor/refcount.h \
	$(top_srcdir)/include/tensor/tensor.h \
	$(top_srcdir)/include/tensor/tools.h \
	$(top_srcdir)/include/tensor/vector.h