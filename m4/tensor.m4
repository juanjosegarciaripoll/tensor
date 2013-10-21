dnl -*- Autoconf -*-
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
    ESSL_XTRA64="-lm -L/opt/ibmcmp/lib64 -lxlf90_r -lxl -lxlomp_ser -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib64 -L/opt/ibmcmp/xlf/${xlf_version}/lib64 -R/opt/ibmcmp/lib64"
    ESSL_XTRA="-lm -L/opt/ibmcmp/lib -lxlf90_r -lxl -lxlomp_ser -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib -L/opt/ibmcmp/xlf/${xlf_version}/lib -R/opt/ibmcmp/lib"
    ESSL_XTRASMP64="-lm -L/opt/ibmcmp/lib64 -lxlf90_r -lxl -lxlsmp -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib64 -L/opt/ibmcmp/xlf/${xlf_version}/lib64 -R/opt/ibmcmp/lib64"
    ESSL_XTRASMP="-lm -L/opt/ibmcmp/lib -lxlf90_r -lxl -lxlsmp -lxlfmath -L/opt/ibmcmp/xlsmp/${xlsmp_version}/lib -L/opt/ibmcmp/xlf/${xlf_version}/lib -R/opt/ibmcmp/lib"
  else
    # AIX version of the libraries
    ESSL_CXXFLAGS="-qnocinc=/usr/include/essl"
  fi
  have_lapack_essl=no
  if test $ac_cv_sizeof_long = 8 ; then    
    AC_CHECK_LIB([esslsmp6464], [esvsgemm],
                 [have_essl=yes;
		  ESSL_F77="xlf_r -q64 -qnosave -qintsize=8";
                  ESSL_LIBS="-llapack_essl6464 -lesslsmp6464 $ESSL_XTRASMP64";
                  ESSL_CXXFLAGS="-D_ESV6464 $ESSL_CXXFLAGS"],
                 [have_essl=no],
                 [$ESSL_XTRASMP64])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl6464], [esvsgemm],
                   [have_essl=yes;
		    ESSL_F77="xlf_r -q64 -qnosave -qintsize=8";
                    ESSL_LIBS="-llapack_essl6464 -lessl6464 $ESSL_XTRA64";
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
		  ESSL_F77="xlf_r -qnosave";
                  ESSL_LIBS="-llapack_essl -lesslsmp $ESSL_XTRASMP"],
                 [have_essl=no],
                 [$ESSL_XTRASMP])
    if test $have_essl = no ; then
      AC_CHECK_LIB([essl], [esvsgemm],
                   [have_essl=yes;
		    ESSL_F77="xlf_r -qnosave";
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
  AC_CHECK_HEADER([mkl.h], [have_mkl=yes], [have_mkl=no])
  AC_MSG_CHECKING([for MKL library])
  if test $have_mkl = yes ; then
    case ${host_cpu} in
      ia64*)    MKL_LIBS="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
      x86_64*)  MKL_LIBS="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
      *)        MKL_LIBS="-lmkl_intel -lmkl_intel_thread -lmkl_core -liomp5 -lpthread";;
    esac
  fi
  AC_MSG_RESULT([$have_mkl])
])

dnl ----------------------------------------------------------------------
dnl Find the MKL library
dnl
AC_DEFUN([TENSOR_CBLAPACK],[
  AC_CHECK_HEADER([cblapack.h], [have_cblapack=yes], [have_cblapack=no])
  AC_MSG_RESULT([$have_cblapack])
  CBLAPACK_LIBS="$LIBS -lcblapack -lf2c"
  CBLAPACK_CXXFLAGS=""
])

dnl ----------------------------------------------------------------------
dnl Choose library
dnl
AC_DEFUN([TENSOR_CHOOSE_LIB],[
TENSOR_VECLIB
TENSOR_ATLAS
TENSOR_MKL
TENSOR_ESSL
TENSOR_CBLAPACK
  if test "x$with_backend" = "x"; then
    with_backend=auto
  fi
  if test "$with_backend" = "auto" -a "$have_mkl" = "yes"; then
    with_backend=mkl
  fi
  if test "$with_backend" = "auto" -a "$have_atlas" = "yes"; then
    with_backend=atlas
  fi
  if test "$with_backend" = "auto" -a "$have_veclib" = "yes"; then
    with_backend=veclib
  fi
  if test "$with_backend" = "auto" -a "$have_cblapack" = "yes"; then
    with_backend=cblapack
  fi
  if test "$with_backend" = "auto" -a "$have_essl" = "yes"; then
    with_backend=essl
  fi
  case "x${with_backend}" in
   xmkl)
    if test $have_mkl = yes; then
      AC_DEFINE(TENSOR_USE_MKL, [1], [Use Intel MKL for matrix operations])
      NUM_LIBS="$LIBS $MKL_LIBS"
      CXXFLAGS="$CXXFLAGS $MKL_CXXFLAGS"
    else
      AC_MSG_ERROR([Intel MKL libraries are not available])
    fi;;
   xveclib)
    if test $have_veclib = yes; then
      AC_DEFINE(TENSOR_USE_VECLIB, [1], [Use VecLib for matrix operations])
      NUM_LIBS="$LIBS $VECLIB_LIBS"
      CXXFLAGS="$CXXFLAGS $VECLIB_CXXFLAGS"
    else
      AC_MSG_ERROR([Apple Veclib libraries are not available])
    fi;;
   xatlas)
    if test $have_atlas = yes; then
      AC_DEFINE(TENSOR_USE_ATLAS, [1], [Use Atlas for matrix operations])
      NUM_LIBS="$LIBS $ATLAS_LIBS"
      CXXFLAGS="$CXXFLAGS $ATLAS_CXXFLAGS"
    else
      AC_MSG_ERROR([Atlas libraries are not available])
    fi;;
   xessl)
    if test $have_essl = yes; then
      AC_DEFINE(TENSOR_USE_ESSL, [1], [Use ESSL for matrix operations])
      F77="$ESSL_F77"
      NUM_LIBS="$LIBS $ESSL_LIBS"
      CXXFLAGS="$CXXFLAGS $ESSL_CXXFLAGS"
    else
      AC_MSG_ERROR([ESSL libraries are not available])
    fi;;
   xcblapack)
    if test $have_cblapack = yes; then
      AC_DEFINE(TENSOR_USE_CBLAPACK, [1], [Use CBLAPACK for matrix operations])
      NUM_LIBS="$LIBS $CBLAPACK_LIBS"
      CXXFLAGS="$CXXFLAGS $CBLAPACK_CXXFLAGS"
    else
      AC_MSG_ERROR([CBLAPACK libraries are not available])
    fi;;
   *)
    AC_MSG_ERROR([No BLAS/LAPACK library available])
  esac
  AM_CONDITIONAL([BUILD_ESSL_LAPACK],
                 [test ${with_backend}${have_essl_lapack} = esslno])
])

dnl ------------------------------------------------------------
dnl f2c code needs to invoke certain functions at boot and exit
dnl
AC_DEFUN([TENSOR_F77_INIT_CODE],[
  OLDLIBS="$LIBS"
  LIBS="$FLIBS $LIBS"
  AC_LINK_IFELSE(
    [AC_LANG_SOURCE([[
      extern void libf2c_init(int argc, char **argv);
      extern void libf2c_close();
      int main(int argc, char **argv) {
        libf2c_init(argc, argv);
	libf2c_close();
	return 0;
      }]])],
    [AC_DEFINE(HAVE_LIBF2C_INIT, [1], [F77 init code])],
    [])
  LIBS="$OLDLIBS"
])

dnl ----------------------------------------------------------------------
dnl Find the Google Test library
dnl
AC_DEFUN([TENSOR_GTEST],[
here=`pwd`
GTEST_URL=http://googletest.googlecode.com/files/gtest-1.6.0.zip
GTEST_NAME=`basename $GTEST_URL .zip`
GTEST_TMP="${here}/test/tmp.zip"
test -d "${here}/test" || mkdir "${here}/test"
AC_MSG_CHECKING([for googletest-read-only in ${ac_confdir}/test])
if test -d "${ac_confdir}/test/googletest-read-only"; then
  AC_MSG_RESULT([yes])
  GTEST_DIR="${ac_confdir}/test/googletest-read-only"
else
  AC_MSG_RESULT([no])
fi
if test "X$GTEST_DIR" = X; then
  AC_MSG_CHECKING([for $GTEST_NAME in ${ac_confdir}/test])	
  if test -d "${ac_confdir}/test/$GTEST_NAME" ; then
    AC_MSG_RESULT([yes])
    GTEST_DIR="${ac_confdir}/test/$GTEST_NAME"
  else
    AC_MSG_RESULT([no])
  fi
fi
if test "X$GTEST_DIR" = X ; then
  AC_MSG_CHECKING([for $GTEST_NAME in build directory]);
  if test -d "${here}/test/$GTEST_NAME" ; then
    AC_MSG_RESULT([yes])
    GTEST_DIR="${here}/test/$GTEST_NAME"
  fi
  if test -d "${here}/$GTEST_NAME" ; then
    GTEST_DIR="${here}/$GTEST_NAME"
  fi
  if test "X$GTEST_DIR" = X ; then
    AC_MSG_RESULT([no])
  else
    AC_MSG_RESULT([yes])
  fi
fi
if test "X$GTEST_DIR" = X ; then
  AC_MSG_CHECKING([trying to download Google Test library])
  if (which unzip && which curl && \
      curl $GTEST_URL > "${GTEST_TMP}" 2>/dev/null && \
      unzip -x "${GTEST_TMP}" -d "${here}/test/" && \
      rm "${GTEST_TMP}") >&AS_MESSAGE_LOG_FD; then
    AC_MSG_RESULT([done])
    GTEST_DIR="${here}/test/$GTEST_NAME"
  else
    if (which unzip && which wget && \
        wget --output-document="${GTEST_TMP}" $GTEST_URL >/dev/null 2>&1 && \
        unzip -x "${GTEST_TMP}" -d "${here}/test/" && \
        rm "${GTEST_TMP}") >&AS_MESSAGE_LOG_FD; then
      AC_MSG_RESULT([done])
      GTEST_DIR="${here}/test/$GTEST_NAME"
    else
      AC_MSG_RESULT([failed])
      GTEST_DIR=""
      AC_MSG_WARN([For testing, please download and unpack google test library]
	          [ $GTEST_URL ]
	          [in ${ac_confdir}/test/]
		  [before configuring tensor])
    fi
  fi
fi
AM_CONDITIONAL([HAVE_GTEST], [test "x${GTEST_DIR}" != x])
AM_CONDITIONAL([TENSOR_THREADSAFE_DEATHTEST], [test "x${enable_threadsafe_deathtest}" == xyes])
AC_SUBST(GTEST_DIR)
])

dnl ----------------------------------------------------------------------
dnl Find the FFTW library
dnl
AC_DEFUN([TENSOR_FFTW],[
  if test "x$with_fftw" = xyes; then
    AC_CHECK_LIB([fftw3], [fftw_plan_dft], [have_fftw=yes], [have_fftw=no])
    AC_MSG_CHECKING([for FFTW library])
    if test $have_fftw = yes -a $with_fftw = yes ; then
      FFTW_LIBS="$LIBS -lfftw3"
      AC_DEFINE([TENSOR_USE_FFTW3], [1], [Use FFTW3 library])
      have_fftw=yes
    else
      have_fftw=no
      with_fftw=no
    fi
    AC_MSG_RESULT([$have_fftw])
  else
    with_fftw=no
  fi
  AM_CONDITIONAL([WITH_FFTW3], [test "x$with_fftw" = xyes])
])

dnl ----------------------------------------------------------------------
dnl Clean up FLIBS removing possible C runtimes
dnl
AC_DEFUN([TENSOR_CLEANUP_FLIBS],[
NEW_FLIBS=""
for i in $FLIBS; do
    if echo $i | egrep 'crt|clang_rt|libm'; then
        eval
    else
       	NEW_FLIBS="$NEW_FLIBS $i"
    fi
done
FLIBS="$NEW_FLIBS"
])