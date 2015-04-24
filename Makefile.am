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

install-doxygen-doc: doxygen-doc
	$(MKDIR_P) $(DESTDIR)$(docdir)/{html,html/search}
	for i in doc/html/*.{html,js,css,png}; do \
		$(INSTALL_DATA) $$i $(DESTDIR)$(docdir)/html/; \
	done
	for i in doc/html/search/*.{js,html,png,css}; do \
		$(INSTALL_DATA) $$i $(DESTDIR)$(docdir)/html/search/; \
	done

uninstall-doxygen-doc:
	rm $(DESTDIR)$(docdir)/html/search/*
	rm $(DESTDIR)$(docdir)/html/*
	-rmdir $(DESTDIR)$(docdir)/html/search
	-rmdir $(DESTDIR)$(docdir)/html
