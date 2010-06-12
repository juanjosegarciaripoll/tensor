PLAT = _essl6464
FORTRAN = xlf_r
OPTS = -q64 -O -qnosave -qintsize=8
DRVOPTS  = $(OPTS)
NOOPT    =
LOADER   = xlf_r
LOADOPTS =
TIMER    = EXT_ETIME_
ARCH     = ar
ARCHFLAGS= cr
RANLIB   = ranlib
BLASLIB      = -lessl6464
LAPACKLIB    = liblapack$(PLAT).a
