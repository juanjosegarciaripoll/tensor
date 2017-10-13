# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = \
	-I$(top_srcdir) -I$(top_srcdir)/include \
	-I$(top_builddir)/include $(F2C_CPPFLAGS)

#
# Main library
#
lib_LTLIBRARIES = libtensor.la

include Makefile.inc

libtensor_la_SOURCES = $(basic_SOURCES)

if WITH_FFTW3
libtensor_la_SOURCES += $(fftw_SOURCES)
endif

AM_FFLAGS = $(FFLAGS)
if TENSOR_ARPACK
libtensor_la_SOURCES += $(arpack_cpp_SOURCES)
if BUILD_ARPACK
libtensor_la_SOURCES +=	$(arpack_SOURCES)
endif
if USE_F2C
libtensor_la_SOURCES += $(arpack_f2c_SOURCES)
AM_FFLAGS += -E
endif
libtensor_la_LIBS = $(FLIBS)
endif

if HAVE_64BITS
libtensor_la_SOURCES += $(rand64_SOURCES)
else
libtensor_la_SOURCES += $(rand32_SOURCES)
endif
