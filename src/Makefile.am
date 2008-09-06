# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include

#
# Main library
#
lib_LTLIBRARIES = libtensor.la

libtensor_la_SOURCES = \
	tools/tictoc.cc \
	tools/refcount.cc \
	rand/rand.cc

if HAVE_64BITS
libtensor_la_SOURCES += rand/rand64.cc
else
libtensor_la_SOURCES += rand/rand32.cc
endif

pkginclude_HEADERS = \
	$(top_srcdir)/include/tensor/tools.h \
	$(top_srcdir)/include/tensor/numbers.h \
	$(top_srcdir)/include/tensor/rand.h
