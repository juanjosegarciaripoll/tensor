cmake_minimum_required(VERSION 3.13)

# Find our extra modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

# Source files are named relative but transformed into absolute paths
cmake_policy(SET CMP0076 NEW)

# set the project name and version
project(tensor
  VERSION 0.2.0
  DESCRIPTION "C++ library for tensors and linear algebra"
  HOMEPAGE_URL https://github.com/juanjosegarciaripoll/tensor)

# specify the C++ standard
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

# Used modules
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(GNUInstallDirs)

############################################################
# SYTEM CHECKS AND CONFIGURATION
#

# Configuration options
option(TENSOR_FFTW "Enable support for FFTW" ON)
option(TENSOR_ARPACK "Enable support for Arpack" OFF)
option(TENSOR_TEST "Enable support for tests with GoogleTest" ON)

# Fortran and Fortran-like libraries
find_package(BLAS)
find_package(LAPACK)

# Search for a C/C++ interface for those libraries
set(CBLAS_FOUND FALSE)
if(${BLAS_LIBRARIES} MATCHES mkl)
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_MKL)
endif()
if(${BLAS_LIBRARIES} MATCHES acml)
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_ACML)
endif()
if(${BLAS_LIBRARIES} MATCHES atlas)
  check_include_file("atlas/cblas.h" HAVE_ATLAS_CBLAS_H)
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_ATLAS TRUE)
endif()
if(${BLAS_LIBRARIES} MATCHES openblas)
  check_include_file("OpenBLAS/cblas.h" HAVE_OPENBLAS_CBLAS_H)
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_OPENBLAS TRUE)
endif()
if(${BLAS_LIBRARIES} MATCHES veclib)
  check_include_file("accelerate/accelerate.h" HAVE_ACCELERATE_ACCELERATE_H)
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_VECLIB TRUE)
endif()
if(NOT ${CBLAS_FOUND})
  check_include_file("cblas.h" HAVE_CBLAS_H)
endif()

# Additional headers and details used by Tensor
check_symbol_exists(dladdr "dlfcn.h" HAVE_DLADDR)
check_symbol_exists(backtrace "execinfo.h" HAVE_BACKTRACE)
check_symbol_exists(backtrace_symbols "execinfo.h" HAVE_BACKTRACE_SYMBOLS)
check_symbol_exists(gettimeofday "gettimeofday.h" HAVE_GETTIMEOFDAY)
include(CheckCSourceCompiles)
check_c_source_compiles("
void *foo() { return __builtin_return_address(1); }
      int main() {
        return (foo() == 0);
      }
" HAVE___BUILTIN_RETURN_ADDRESS)
check_type_size(long SIZEOF_LONG)
if(${SIZEOF_LONG} EQUAL 8)
  set(TENSOR_64BITS TRUE)
else()
  set(TENSOR_64BITS FALSE)
endif()

# Export configuration file
configure_file(include/tensor/config.h.cmake.in ${CMAKE_BINARY_DIR}/include/tensor/config.h)

############################################################
# LIBRARY
#

add_library(tensor STATIC "")
target_include_directories(tensor
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE lib
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/config_aux>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  )
target_link_options(tensor
  PUBLIC ${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS})
target_link_libraries(tensor
  PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
add_subdirectory(include)
add_subdirectory(src)

############################################################
# TESTING
#
if(${TENSOR_TEST})
  add_subdirectory(tests)
endif()

############################################################
# INSTALLATION
#
install(TARGETS tensor
  EXPORT tensorConfig
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/tensorConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
  )
install(EXPORT tensorConfig
  DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/tensor"
  )
install(FILES ${CMAKE_BINARY_DIR}/tensorConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/tensor"
  )
