cmake_minimum_required(VERSION 3.14)
message(STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")

# Source files are named relative but transformed into absolute paths
cmake_policy(SET CMP0076 NEW)

# set the project name and version
project(tensor
  VERSION 0.2.0
  DESCRIPTION "C++ library for tensors and linear algebra"
  HOMEPAGE_URL https://github.com/juanjosegarciaripoll/tensor)

# specify the C++ standard
# GoogleTest requiers C++11 or better
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

# Used modules
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(GNUInstallDirs)
include(TestBigEndian)
find_package(PkgConfig QUIET)

# Find our extra modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(TensorDependencies)

############################################################
# SYTEM CHECKS AND CONFIGURATION
#

# Configuration options
option(TENSOR_FFTW "Enable support for FFTW" OFF)
option(TENSOR_ARPACK "Enable support for Arpack" OFF)
option(TENSOR_TEST "Enable support for tests with GoogleTest" ON)

# Fortran and Fortran-like libraries
message(STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
# find_package(TensorBLAS REQUIRED)

# Search for a C/C++ interface for those libraries
set(CBLAS_FOUND FALSE)
if(TENSOR_BLAS MATCHES MKL)
  find_package(MKL)
  if(NOT MKL_FOUND)
	message(FATAL_ERROR "MKL library not found")
  endif()
  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_MKL TRUE)
endif()
if((TENSOR_BLAS MATCHES OpenBLAS) OR (NOT CBLAS_FOUND))
  tensor_find_dependency(
    NAME OpenBLAS
    CONFIG_NAME OpenBLAS
    IMPORT_TARGET OpenBLAS::OpenBLAS
    PKGCONFIG_NAMES openblas openblas64
  )

  set(CBLAS_FOUND TRUE)
  set(TENSOR_USE_OPENBLAS TRUE)
  set(HAVE_CBLAS_H TRUE)
endif()
if(NOT CBLAS_FOUND)
  message(FATAL_ERROR "Cannot build Tensor library")
endif()
if(NOT TENSOR_USE_MKL)
  find_package(LAPACK)
  if(NOT LAPACK_FOUND)
	message(FATAL_ERROR "LAPACK library not found")
  endif()
  set(TENSOR_USE_LAPACK TRUE)
endif()
#
# Fortran name mangling
#
if(TENSOR_ARPACK)
  find_package(ARPACK QUIET)
  set(ARPACK_TARGET ARPACK::ARPACK)
  if(NOT ARPACK_FOUND)
     tensor_find_dependency(
       NAME arpack
       CONFIG_NAME arpack-ng
       IMPORT_TARGET ARPACK::ARPACK
       PKGCONFIG_NAMES arpack-ng arpack
     )

     set(ARPACK_TARGET Dependency::arpack)
	set(ARPACK_FOUND TRUE)
  endif()

  if(NOT ARPACK_FOUND)
	message(FATAL_ERROR "Unable to build with ARPACK though requested")
  endif()

  enable_language(Fortran)
  include(FortranCInterface)
  FortranCInterface_HEADER(dummy.h)
  if(FortranCInterface_GLOBAL_FOUND)
    if(${FortranCInterface_GLOBAL_CASE} STREQUAL "UPPER")
      set(F77_FUNC NAME)
    else()
      set(F77_FUNC name)
    endif()
    if(NOT ("${FortranCInterface_GLOBAL_PREFIX}" STREQUAL ""))
      set(F77_FUNC "${FortranCInterface_GLOBAL_PREFIX}##${F77_FUNC}")
    endif()
    if(NOT ("${FortranCInterface_GLOBAL_SUFFIX}" STREQUAL ""))
      set(F77_FUNC "${F77_FUNC}##${FortranCInterface_GLOBAL_SUFFIX}")
    endif()
  else()
    message(FATAL_ERROR "Unable to find Fortran-C interface")
  endif()
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
    CONFIG_NAME FFTW3
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
target_include_directories(tensor
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/config_aux>
  )
if(TENSOR_ARPACK AND ARPACK_FOUND)
  target_link_libraries(tensor PUBLIC ${ARPACK_TARGET})
endif()
if(TENSOR_USE_OPENBLAS)
  target_link_libraries(tensor PRIVATE Dependency::OpenBLAS)
endif()
if(TENSOR_USE_LAPACK)
  target_link_libraries(tensor PRIVATE LAPACK::LAPACK)
endif()
if(TENSOR_USE_MKL)
  target_link_libraries(tensor PRIVATE MKL::MKL)
endif()
if(TENSOR_FFTW)
  target_link_libraries(tensor PRIVATE Dependency::fftw3)
endif()
add_subdirectory(include)
add_subdirectory(src)

############################################################
# TESTING
#
if(TENSOR_TEST)
  find_package(GTest)
  set(GTEST_PREFIX GTest::)
  if(NOT GTest_FOUND)
	include(FetchContent)
	FetchContent_Declare(
	  googletest
	  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
	  )
	# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
	set(GTEST_PREFIX "")
  endif()
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
  ${PROJECT_BINARY_DIR}/tensorConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
  )
install(EXPORT tensorConfig
  NAMESPACE tensor::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tensor"
  )
install(FILES ${PROJECT_BINARY_DIR}/tensorConfigVersion.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tensor"
  )
