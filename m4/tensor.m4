dnl ----------------------------------------------------------------------
dnl Find veclib framework
dnl
AC_DEFUN([TENSOR_VECLIB],[
  AC_MSG_CHECKING([for VecLib library])
  if test `uname` = "Darwin"; then
    have_veclib=yes
    VECLIB_LIBS="-framework veclib"
    VECLIB_CXXFLAGS="-framework veclib"
  else
    have_veclib=no
  fi
  AC_MSG_RESULT([$have_veclib])
])

dnl ----------------------------------------------------------------------
dnl Find the ATLAS library
dnl
AC_DEFUN([TENSOR_ATLAS],[
  AC_CHECK_LIB([atlas], [ATL_buildinfo], [have_atlas=yes], [have_atlas=no])
  AC_MSG_CHECKING([for ATLAS library])
  if test $have_atlas = yes ; then
    ATLAS_LIBS="$LIBS -llapack -lcblas -latlas"
    ATLAS_CXXFLAGS=""
  fi
  AC_MSG_RESULT([$have_atlas])
])


dnl ----------------------------------------------------------------------
dnl Find the MKL library
dnl
AC_DEFUN([TENSOR_MKL],[
  OLD_LDFLAGS="$LDFLAGS"
  if test -d $MKL_DIR ; then
    case ${host_cpu} in
      ia64*)    MKL_LIBS="-L$MKL_DIR/lib/ia64";;
      x86_64*)  MKL_LIBS="-L$MKL_DIR/lib/64";;
      *)        MKL_LIBS="-L$MKL_DIR/lib/32";;
    esac
    MKL_CXXFLAGS="-I$MKL_DIR/include"
    LDFLAGS="$LDFLAGS $MKL_LIBS"
  fi
  AC_CHECK_LIB([mkl_core], [mkl_blas_caxpy], [have_mkl=yes], [have_mkl=no])
  AC_MSG_CHECKING([for MKL library])
  if test $have_mkl = yes ; then
    case ${host_cpu} in
      ia64*)    MKL_LIBS="$MKL_LIBS -lmkl_intel_lp64 -lmkl_core -lpthread";;
      x86_64*)  MKL_LIBS="$MKL_LIBS -lmkl_intel_lp64 -lmkl_core -lpthread";;
      *)        MKL_LIBS="$MKL_LIBS -lmkl_intel -lmkl_core -lpthread";;
    esac
  fi
  AC_MSG_RESULT([$have_mkl])
  LDFLAGS="$OLD_LDFLAGS"
])

dnl ----------------------------------------------------------------------
dnl Choose library
dnl
AC_DEFUN([TENSOR_CHOOSE_LIB],[
  TENSOR_VECLIB
  TENSOR_ATLAS
  TENSOR_MKL
  if test $have_mkl = yes; then
    AC_DEFINE(TENSOR_USE_MKL, [1], [Use Intel MKL for matrix operations])
    LIBS="$LIBS $MKL_LIBS"
    CXXFLAGS="$CFLAGS $MKL_CXXFLAGS"
    have_atlas=no
    have_veclib=no
  fi
  if test $have_veclib = yes; then
    AC_DEFINE(TENSOR_USE_VECLIB, [1], [Use VecLib for matrix operations])
    LIBS="$LIBS $MKL_LIBS"
    CXXFLAGS="$CFLAGS $MKL_CXXFLAGS"
    have_atlas=no
  fi
  if test $have_atlas = yes; then
    AC_DEFINE(TENSOR_USE_ATLAS, [1], [Use Atlas for matrix operations])
    LIBS="$LIBS $ATLAS_LIBS"
    CXXFLAGS="$CFLAGS $ATLAS_CXXFLAGS"
  fi
])