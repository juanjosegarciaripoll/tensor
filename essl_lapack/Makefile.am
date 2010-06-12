#
# ESSL/LAPACK interfaces
#

lib_LIBRARIES = $(ESSL_LAPACK_LIB)

EXTRA_LIBRARIES = liblapack_essl.a liblapack_essl64.a liblapack_essl6464.a

liblapack_essl.a: lapack-lite/liblapack_essl.a
	cp lapack-lite/liblapack_essl.a $@
lapack-lite/liblapack_essl.a: $(srcdir)/make32.inc lapack-lite/Makefile
	cp $(srcdir)/make32.inc lapack-lite/make.inc && \
	cd lapack-lite && $(MAKE) clean && $(MAKE) lapacklib

liblapack_essl64.a: lapack-lite/liblapack_essl64.a
	cp lapack-lite/liblapack_essl64.a $@
lapack-lite/liblapack_essl64.a: $(srcdir)/make64.inc lapack-lite/Makefile
	cp $(srcdir)/make64.inc lapack-lite/make.inc && \
	cd lapack-lite && $(MAKE) clean && $(MAKE) lapacklib

liblapack_essl6464.a: lapack-lite/liblapack_essl6464.a
	cp lapack-lite/liblapack_essl6464.a $@
lapack-lite/liblapack_essl6464.a: $(srcdir)/make6464.inc lapack-lite/Makefile
	cp $(srcdir)/make6464.inc lapack-lite/make.inc && \
	cd lapack-lite && $(MAKE) clean && $(MAKE) lapacklib

lapack-lite/Makefile: lapack-lite-3.1.1.tgz LAPACK/ccisrc.tar
	tar xzf $<
	mv lapack-lite-3.1.1 lapack-lite
	cd lapack-lite && tar xf ../LAPACK/ccisrc.tar
	touch $@
LAPACK/ccisrc.tar: cci.tgz
	tar xzf $<
lapack-lite-3.1.1.tgz:
	curl http://www.netlib.org/lapack/lapack-lite-3.1.1.tgz > $@
cci.tgz:
	curl http://www.netlib.org/lapack/essl/cci.tgz > $@
