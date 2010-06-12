dnl ----------------------------------------------------------------------
dnl Find word size
dnl
AC_DEFUN([TENSOR_BITS],[
  AC_CHECK_SIZEOF([long])
  AM_CONDITIONAL([HAVE_64BITS], [test $ac_cv_sizeof_long = 8])
  if test $ac_cv_sizeof_long = 8; then
    AC_DEFINE(TENSOR_64BITS, [1], [Words are 64-bits])
  fi
])


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
dnl Find the ESSL library
dnl
AC_DEFUN([TENSOR_ESSL],[
  AC_MSG_CHECKING([for ESSL library])
  if test $ac_cv_sizeof_long = 8
    
    AC_CHECK_LIB([esslsmp6464], [esvsgemm],
                 [have_essl=yes;
                  ESSL_LIBS=-lesslsmp6464;
                  ESSL_CXXFLAGS=-D_ESV6464],
                 [have_essl=no])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl], [esvsgemm],
                   [have_essl=yes;
                    ESSL_LIBS=-lessl6464
                    ESSL_CXXFLAGS=-D_ESV6464],
                   [have_essl=no])
    fi
  else
    AC_CHECK_LIB([esslsmp], [esvsgemm],
                 [have_essl=yes; ESSL_LIBS=-lesslsmp], [have_essl=no])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl], [esvsgemm],
                   [have_essl=yes; ESSL_LIBS=-lessl], [have_essl=no])
    fi
  fi
  LIBS="$LIBS $ESSLIB"
  AC_MSG_RESULT([$have_essl])
])


dnl ----------------------------------------------------------------------
dnl Find the MKL library
dnl
AC_DEFUN([TENSOR_MKL],[
  OLD_LDFLAGS="$LDFLAGS"
  if test -d $MKL_DIR ; then
    case ${host_cpu} in
      ia64*)    MKL_LIBS="-L$MKL_DIR/lib/64";;
      x86_64*)  MKL_LIBS="-L$MKL_DIR/lib/emt64";;
      *)        MKL_LIBS="-L$MKL_DIR/lib/32";;
    esac
    MKL_CXXFLAGS="-I$MKL_DIR/include"
    LDFLAGS="$LDFLAGS $MKL_LIBS"
  fi
  AC_CHECK_LIB([mkl_core], [mkl_lapack_cbdsqr], [have_mkl=yes], [have_mkl=no])
  AC_MSG_CHECKING([for MKL library])
  if test $have_mkl = yes ; then
    case ${host_cpu} in
      ia64*)    MKL_LIBS="$MKL_LIBS -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
      x86_64*)  MKL_LIBS="$MKL_LIBS -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
      *)        MKL_LIBS="$MKL_LIBS -lmkl_intel -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
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
    CXXFLAGS="$CXXFLAGS $MKL_CXXFLAGS"
    have_atlas=no
    have_veclib=no
  fi
  if test $have_veclib = yes; then
    AC_DEFINE(TENSOR_USE_VECLIB, [1], [Use VecLib for matrix operations])
    LIBS="$LIBS $VECLIB_LIBS"
    CXXFLAGS="$CXXFLAGS $VECLIB_CXXFLAGS"
    have_atlas=no
  fi
  if test $have_atlas = yes; then
    AC_DEFINE(TENSOR_USE_ATLAS, [1], [Use Atlas for matrix operations])
    LIBS="$LIBS $ATLAS_LIBS"
    CXXFLAGS="$CXXFLAGS $ATLAS_CXXFLAGS"
    have_essl=no
  fi
  if test $have_essl = yes; then
    AC_DEFINE(TENSOR_USE_ESSL, [1], [Use ESSL for matrix operations])
    LIBS="$LIBS $ESSL_LIBS"
    CXXFLAGS="$CXXFLAGS $ESSL_CXXFLAGS"
  fi
])