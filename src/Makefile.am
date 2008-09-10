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
	tensor/tensor_z.cc

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