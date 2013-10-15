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
	$(MKDIR_P) $(DESTDIR)$(docdir)/{html,html/search}
	for i in doc/html/*.{html,js,css,png} doc/html/search/*.{js,html}; do \
		$(INSTALL_DATA) $$i $(DESTDIR)$(docdir)/html; \
	done

uninstall-doxygen-doc:
	rm $(DESTDIR)$(docdir)/html/search/*
	rm $(DESTDIR)$(docdir)/html/*
	-rmdir $(DESTDIR)$(docdir)/html/search
	-rmdir $(DESTDIR)$(docdir)/html
