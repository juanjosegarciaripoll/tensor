PLAT = _essl6464
FORTRAN = xlf_r -q64 -qnosave -qintsize=8
OPTS = -O
DRVOPTS  = $(OPTS)
NOOPT    =
LOADER   = $(FORTRAN)
LOADOPTS = $(OPTS)
TIMER    = EXT_ETIME_
ARCH     = ar
ARCHFLAGS= cr
RANLIB   = ranlib
BLASLIB      = -lessl6464
LAPACKLIB    = liblapack$(PLAT).a
