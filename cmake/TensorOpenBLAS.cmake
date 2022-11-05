include_guard()

include(CheckCSourceCompiles)

function(tensor_openblas_requires_lapack)
  cmake_parse_arguments(ARGP "" "TARGET;OUTVAR" "" ${ARGN})
  set(CMAKE_REQUIRED_LIBRARIES ${ARGP_TARGET})
  check_c_source_compiles(
    "int main() { extern void dgesvd(); dgesvd(); }"
    OPENBLAS_HAS_DGESVD
  )
  if (NOT OPENBLAS_HAS_DGESVD)
    check_c_source_compiles(
        "int main() { extern void dgesvd_(); dgesvd_(); }"
        OPENBLAS_HAS_DGESVD_
    )
    if (NOT OPENBLAS_HAS_DGESVD_)
      message(STATUS "OpenBLAS requires separate LAPACK library")
      set(${ARGP_OUTVAR} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  message(STATUS "OpenBLAS does not require separate LAPACK library")
  set(${ARGP_OUTVAR} FALSE PARENT_SCOPE)
endfunction()

