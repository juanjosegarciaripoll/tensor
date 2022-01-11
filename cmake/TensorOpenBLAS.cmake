include_guard()

include(CheckCSourceCompiles)

function(tensor_openblas_requires_lapack)
  cmake_parse_arguments(ARGP "" "TARGET" "" ${ARGN})
  set(CMAKE_REQUIRED_LIBRARIES ${TARGET})
  check_c_source_compiles(
    "int main() { extern void *dsgevd; if (dsgevd != 0) return 0; return 1; }"
    OPENBLAS_HAS_LAPACK
  )
  if (NOT OPENBLAS_HAS_LAPACK)
    check_c_source_compiles(
        "int main() { extern void *dsgevd_; if (dsgevd_ != 0) return 0; return 1; }"
        OPENBLAS_HAS_LAPACK
    )
    if (NOT OPENBLAS_HAS_LAPACK)
      message(STATUS "OpenBLAS requires separate LAPACK library")
      set(OPENBLAS_REQUIRES_LAPACK TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
  message(STATUS "OpenBLAS does not require separate LAPACK library")
  set(OPENBLAS_REQUIRES_LAPACK FALSE PARENT_SCOPE)
endfunction()

