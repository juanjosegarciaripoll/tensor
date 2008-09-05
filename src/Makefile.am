# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include

#
# Main library
#
lib_LTLIBRARIES = libtensor.la

libtensor_la_SOURCES = \
	tools/tictoc.cpp

pkginclude_HEADERS = \
	$(top_srcdir)/include/tensor/tools.h
