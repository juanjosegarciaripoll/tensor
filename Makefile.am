ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include www scripts

include aminclude.am

EXTRA_DIST = autogen.sh $(DX_CONFIG) doc/html

if HAVE_GTEST
SUBDIRS += test
endif
if BUILD_ESSL_LAPACK
SUBDIRS += essl_lapack
endif

upload-doc: doxygen-doc
	rsync -rauvz doc/html dreamhost:tream.dreamhosters.com/tensor/
