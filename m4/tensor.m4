dnl ----------------------------------------------------------------------
dnl Find word size
dnl
AC_DEFUN([TENSOR_BITS],[
  AC_CHECK_SIZEOF([long])
  AM_CONDITIONAL([HAVE_64BITS], [test $ac_cv_sizeof_long = 8])
  if test $ac_cv_sizeof_long = 8; then
    AC_DEFINE(TENSOR_64BITS, [1], [Words are 64-bits])
  fi
  AC_C_BIGENDIAN([AC_DEFINE(TENSOR_BIGENDIAN, [1], [Machine is big endian])],[],[])
])

dnl ------------------------------------------------------------
dnl Backtraces
dnl
AC_DEFUN([TENSOR_BACKTRACE],[
  AC_CHECK_FUNCS( [dladdr backtrace backtrace_symbols] )
  AC_CHECK_HEADERS_ONCE( [dlfnc.h] )
  AC_RUN_IFELSE(
    [AC_LANG_SOURCE([[
      void *foo() { return __builtin_return_address(1); }
      int main() {
        return (foo() == 0);
      }]])],
    [AC_DEFINE(HAVE___BUILTIN_RETURN_ADDRESS, [1], [GCC builtin return address])],
    [])
])

dnl ----------------------------------------------------------------------
dnl Find veclib framework
dnl
AC_DEFUN([TENSOR_VECLIB],[
  AC_MSG_CHECKING([for VecLib library])
  if test `uname` = "Darwin"; then
    have_veclib=yes
    VECLIB_LIBS="-framework veclib"
    AC_CHECK_HEADERS_ONCE([vecLib/cblas.h])
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
    AC_CHECK_HEADERS_ONCE([atlas/cblas.h cblas.h])
  fi
  AC_MSG_RESULT([$have_atlas])
])


dnl ----------------------------------------------------------------------
dnl Find the ESSL library
dnl
AC_DEFUN([TENSOR_ESSL],[
  if test -d /opt/ibmcmp; then
    # Linux version of the libraries
    xlsmp_version=`ls /opt/ibmcmp/xlsmp/|tail -1`
    xlf_version=`ls /opt/ibmcmp/xlf/|tail -1`
    ESSL_XTRA64="-lxlf90_r -lxlomp_ser -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib64 -L/opt/ibmcmp/xlf/${xlf_version}/lib64 -R/opt/ibmcmp/lib64"
    ESSL_XTRA="-lxlf90_r -lxlomp_ser -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib -L/opt/ibmcmp/xlf/${xlf_version}/lib -R/opt/ibmcmp/lib"
    ESSL_XTRASMP64="-lxlf90_r -lxlsmp -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib64 -L/opt/ibmcmp/xlf/${xlf_version}/lib64 -R/opt/ibmcmp/lib64"
    ESSL_XTRASMP="-lxlf90_r -lxlsmp -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib -L/opt/ibmcmp/xlf/${xlf_version}/lib -R/opt/ibmcmp/lib"
  else
    # AIX version of the libraries
    ESSL_CXXFLAGS="-qnocinc=/usr/include/essl"
  fi
  have_lapack_essl=no
  if test $ac_cv_sizeof_long = 8 ; then    
    AC_CHECK_LIB([esslsmp6464], [esvsgemm],
                 [have_essl=yes;
                  ESSL_LIBS="-llapack_essl6464 -lesslsmp6464 $ESSL_XTRASMP64";
                  ESSL_CXXFLAGS="-D_ESV6464 $ESSL_CXXFLAGS"],
                 [have_essl=no],
                 [$ESSL_XTRASMP64])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl6464], [esvsgemm],
                   [have_essl=yes;
                    ESSL_LIBS="-llapack_essl6464 -lessl6464 $ESSL_XTRA64"
                    ESSL_CXXFLAGS="-D_ESV6464 $ESSL_CXXFLAGS"],
                   [have_essl=no],
                   [$ESSL_XTRA64])
    fi
    if test $have_essl = yes ; then
      AC_CHECK_LIB([lapack_essl6464], [sgbsv], [have_essl_lapack=yes],
                   [have_essl_lapack=no], [$ESSL_LIBS])
      ESSL_LAPACK_LIB=liblapack_essl6464.a
      AC_SUBST([ESSL_LAPACK_LIB])
    fi
  else
    AC_CHECK_LIB([esslsmp], [esvsgemm],
                 [have_essl=yes;
                  ESSL_LIBS="-llapack_essl -lesslsmp $ESSL_XTRASMP"],
                 [have_essl=no],
                 [$ESSL_XTRASMP])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl], [esvsgemm],
                   [have_essl=yes;
                    ESSL_LIBS="-llapack_essl -lessl $ESSL_XTRA"],
                   [have_essl=no],
                   [$ESSL_XTRA])
    fi
    if test $have_essl = yes ; then
      AC_CHECK_LIB([lapack_essl], [sgbsv], [have_essl_lapack=yes], [have_essl_lapack=no],
                   [$ESSL_LIBS])
      ESSL_LAPACK_LIB=liblapack_essl.a
      AC_SUBST([ESSL_LAPACK_LIB])
    fi
  fi
  AC_MSG_CHECKING([for ESSL library])
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
  TENSOR_ESSL
  if test $have_mkl = yes; then
    AC_DEFINE(TENSOR_USE_MKL, [1], [Use Intel MKL for matrix operations])
    LIBS="$LIBS $MKL_LIBS"
    CXXFLAGS="$CXXFLAGS $MKL_CXXFLAGS"
    have_atlas=no
    have_veclib=no
    have_essl=no
  fi
  if test $have_veclib = yes; then
    AC_DEFINE(TENSOR_USE_VECLIB, [1], [Use VecLib for matrix operations])
    LIBS="$LIBS $VECLIB_LIBS"
    CXXFLAGS="$CXXFLAGS $VECLIB_CXXFLAGS"
    have_atlas=no
    have_essl=no
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
  AM_CONDITIONAL([BUILD_ESSL_LAPACK],
                 [test ${have_essl}${have_essl_lapack} = yesno])
])