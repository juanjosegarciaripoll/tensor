#
# ESSL/LAPACK interfaces
#

lib_LIBRARIES = liblapack_essl.a liblapack_essl64.a liblapack_essl6464.a

liblapack_essl.a: lapack-lite/liblapack_essl.a
	cp lapack-lite/liblapack_essl.a $@
lapack-lite/liblapack_essl.a: make32.inc lapack-lite/Makefile
	cp $(srcdir)/make32.inc lapack-lite/make.inc && \
	cd lapack-lite && \
	$(MAKE) lapacklib

liblapack_essl64.a: liblapack_essl.a lapack-lite/liblapack_essl64.a
	cp lapack-lite/liblapack_essl64.a $@
lapack-lite/liblapack_essl64.a: make64.inc lapack-lite/Makefile
	cp $(srcdir)/make64.inc lapack-lite/make.inc && \
	cd lapack-lite && \
	$(MAKE) lapacklib

liblapack_essl6464.a: liblapack_essl64.a lapack-lite/liblapack_essl6464.a
	cp lapack-lite/liblapack_essl6464.a $@
lapack-lite/liblapack_essl6464.a: make6464.inc lapack-lite/Makefile
	cp $(srcdir)/make6464.inc lapack-lite/make.inc && \
	cd lapack-lite && \
	$(MAKE) lapacklib

lapack-lite/Makefile: lapack-lite-3.1.1.tgz LAPACK
	tar xzf $<
	mv lapack-lite-3.1.1 lapack-lite
	cd lapack-lite && tar xf ../LAPACK/ccisrc.tar
LAPACK: cci.tgz
	tar xzf $<
lapack-lite-3.1.1.tgz:
	curl http://www.netlib.org/lapack/lapack-lite-3.1.1.tgz > $@
cci.tgz:
	curl http://www.netlib.org/lapack/essl/cci.tgz > $@
