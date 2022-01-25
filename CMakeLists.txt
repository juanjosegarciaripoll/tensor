cmake_minimum_required(VERSION 3.18)

# If not set by the user, make a release build.
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Source files are named relative but transformed into absolute paths
cmake_policy(SET CMP0076 NEW)

# set the project name and version
project(tensor
  VERSION 0.2.0
  DESCRIPTION "C++ library for tensors and linear algebra"
  HOMEPAGE_URL https://github.com/juanjosegarciaripoll/tensor)

# specify the C++ standard
# GoogleTest requiers C++14 or better
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Used modules
include(FetchContent)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(GNUInstallDirs)
include(TestBigEndian)
find_package(PkgConfig QUIET)

# Find our extra modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(TensorOptions)
include(TensorDependencies)
include(TensorOpenBLAS)

############################################################
# SYTEM CHECKS AND CONFIGURATION
#

# Configuration options
option(TENSOR_FFTW "Enable support for FFTW" OFF)
option(TENSOR_ARPACK "Enable support for Arpack" OFF)
option(TENSOR_TEST "Enable support for tests with GoogleTest" ON)
option(TENSOR_BENCHMARK "Enable benchmarking code after building" OFF)
set(TENSOR_BLAS "OpenBLAS" CACHE STRING "BLAS backend to use. Values are OpenBLAS (default) or MKL (also brings LAPACK backend)")

# Compiler settings and other tooling
make_tensor_options()

# Search for dependencies
set(TENSOR_DEPENDENCIES)
set(CBLAS_FOUND FALSE)
set(TENSOR_USE_LAPACK TRUE)
if(TENSOR_BLAS STREQUAL MKL)
  tensor_find_dependency(
    NAME MKL
    VAR TENSOR_BLAS_TARGET
    CONFIG_NAMES MKL
    IMPORT_TARGET MKL::MKL
    DEPENDENCIES TENSOR_DEPENDENCIES
    REQUIRED
  )
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_MKL TRUE)
  set(TENSOR_USE_LAPACK FALSE)
endif()
if((TENSOR_BLAS MATCHES OpenBLAS) OR (NOT CBLAS_FOUND))
  tensor_find_dependency(
    NAME OpenBLAS
    VAR TENSOR_BLAS_TARGET
    CONFIG_NAMES OpenBLAS
    IMPORT_TARGET OpenBLAS::OpenBLAS
    DEPENDENCIES TENSOR_DEPENDENCIES
    PKGCONFIG_NAMES openblas openblas64
    REQUIRED
  )
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_OPENBLAS TRUE)
  set(HAVE_CBLAS_H TRUE)
  tensor_openblas_requires_lapack(TARGET ${TENSOR_BLAS_TARGET})
  set(TENSOR_USE_LAPACK ${OPENBLAS_REQUIRES_LAPACK})
else()
  message(FATAL_ERROR "TENSOR_BLAS must be one of MKL or OpenBLAS. Current value is \"${TENSOR_BLAS}\"")
endif()
if(TENSOR_USE_LAPACK)
  tensor_find_dependency(
    NAME lapack
    VAR TENSOR_LAPACK_TARGET
    CONFIG_NAMES LAPACK
    IMPORT_TARGET LAPACK::LAPACK
    DEPENDENCIES TENSOR_DEPENDENCIES
    REQUIRED)
  set(TENSOR_USE_LAPACK TRUE)
endif()
#
# Fortran name mangling
#
if(TENSOR_ARPACK)
  tensor_find_dependency(
    NAME arpack
    VAR TENSOR_ARPACK_TARGET
    CONFIG_NAMES ARPACK arpack-ng
    IMPORT_TARGET ARPACK::ARPACK
    DEPENDENCIES TENSOR_DEPENDENCIES
    PKGCONFIG_NAMES arpack-ng arpack
    REQUIRED)
endif()

#
# Type sizes and endianness
#
TEST_BIG_ENDIAN(TENSOR_BIGENDIAN)

############################################################
# CONFIGURATION FILES

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
if(TENSOR_FFTW)
  tensor_find_dependency(
    NAME fftw3
    VAR TENSOR_FFTW_TARGET
    CONFIG_NAMES FFTW3
    DEPENDENCIES TENSOR_DEPENDENCIES
    IMPORT_TARGET FFTW3::fftw3
    PKGCONFIG_NAMES fftw3
  )
  set(TENSOR_USE_FFTW3 TRUE)
endif()
configure_file(include/config.h.cmake.in ${PROJECT_BINARY_DIR}/include/tensor/config.h)

############################################################
# LIBRARY
#

add_library(tensor "")
target_link_libraries(tensor PRIVATE tensor_options)
target_include_directories(tensor
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/config_aux>
  )
target_link_libraries(tensor PUBLIC ${TENSOR_BLAS_TARGET})
if(TENSOR_USE_LAPACK)
  target_link_libraries(tensor PRIVATE ${TENSOR_LAPACK_TARGET})
endif()
if(TENSOR_ARPACK)
  target_link_libraries(tensor PUBLIC ${TENSOR_ARPACK_TARGET})
endif()
if(TENSOR_FFTW)
  target_link_libraries(tensor PRIVATE ${TENSOR_FFTW_TARGET})
endif()
add_subdirectory(include)
add_subdirectory(src)

############################################################
# PROFILING
#
add_subdirectory("profile")

############################################################
# TESTING
#
if(TENSOR_TEST)
  find_package(GTest)
  set(GTEST_COMPONENT GTest::gtest)
  if(NOT GTest_FOUND)
    message(STATUS "GTest not found")
	  FetchContent_Declare(
	    googletest
	    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
	    )
	  # For Windows: Prevent overriding the parent project's compiler/linker settings
	  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    message(STATUS "Downloading googletest")
	  FetchContent_MakeAvailable(googletest)
	  set(GTEST_COMPONENT "gtest_main")
  endif()
  add_subdirectory(tests)
endif()

############################################################
# INSTALLATION
#
install(TARGETS tensor tensor_options
  EXPORT tensorTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

include(CMakePackageConfigHelpers)
message(STATUS "Tensor installed dependencies:\n${TENSOR_DEPENDENCIES}")
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/tensorConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
  )
install(EXPORT tensorTargets
  NAMESPACE tensor::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tensor"
  )
configure_file(cmake/tensorConfig.cmake.in tensorConfig.cmake @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/tensorConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/tensorConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tensor"
  )
