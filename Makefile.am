ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src

include aminclude.am

EXTRA_DIST = autogen.sh $(DX_CONFIG) doc/html

if HAVE_GTEST
SUBDIRS += test
endif
