dnl ----------------------------------------------------------------------
dnl Find veclib framework
dnl
AC_DEFUN([TENSOR_VECLIB],[
  AC_MSG_CHECKING([for VecLib library])
  if test `uname` = "Darwin"; then
    AC_DEFINE(TENSOR_USE_VECLIB, [1], [Use VecLib for matrix operations])
    have_veclib=yes
    LIBS="$LIBS -framework veclib"
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
    if test $have_veclib = no ; then
      AC_DEFINE(TENSOR_USE_ATLAS, [1], [Use ATLAS for matrix operations])
      LIBS="$LIBS -llapack -lcblas -latlas"
    else
      have_atlas=no
    fi
  fi
  AC_MSG_RESULT([$have_atlas])
])
