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

install-doxygen-doc: doxygen-doc
	$(MKDIR_P) $(DESTDIR)$(docdir)/$(PACKAGE)/{html,html/search}
	for i in doc/html/* doc/html/search/*; do \
		$(INSTALL_DATA) $$i $(DESTDIR)$(docdir)/$(PACKAGE)/html; \
	done

uninstall-doxygen-doc:
	rm $(DESTDIR)$(docdir)/$(PACKAGE)/html/*
	-rmdir $(DESTDIR)$(docdir)/$(PACKAGE)/html
