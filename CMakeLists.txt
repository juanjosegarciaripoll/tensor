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
# GoogleTest requires C++14 or better
set(TENSOR_CXX_STANDARD 14 CACHE STRING
  "Numeric value (14, 17, 20) defining the C++ standard to use")
string(REGEX MATCH "\(14|17|20\)" CMAKE_CXX_STANDARD ${TENSOR_CXX_STANDARD})
if(NOT ${CMAKE_CXX_STANDARD} STREQUAL ${TENSOR_CXX_STANDARD})
  message(SEND_ERROR "Illegal value of TENSOR_CXX_STANDARD ${TENSOR_CXX_STANDARD}, not in 14, 17 or 20")
endif()
set(CMAKE_CXX_STANDARD ${TENSOR_CXX_STANDARD})
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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TENSOR_DEBUG_DEFAULT_VALUE ON)
else()
  set(TENSOR_DEBUG_DEFAULT_VALUE OFF)
endif()

# Configuration options
option(TENSOR_DEBUG "Include assertions for out-of-range access and other conditions" ${TENSOR_DEBUG_DEFAULT_VALUE})
option(TENSOR_FFTW "Enable support for FFTW" OFF)
option(TENSOR_ARPACK "Enable support for Arpack" OFF)
option(TENSOR_PRIMME "Enable support for Primme" OFF)
option(TENSOR_TEST "Enable support for tests with GoogleTest" ON)
option(TENSOR_BENCHMARK "Enable benchmarking code after building" OFF)
option(TENSOR_RANGE_SQUEEZE "Selecting ranges from a tensor can remove dimensions" ON)
option(TENSOR_COPY_ON_WRITE "Tensor data is stored and shared using a copy-on-write model" ON)
option(TENSOR_FORCE_STATIC "Ignore BUILD_SHARED_LIBS forcefully")
option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(TENSOR_BUILD_PROFILE "Build profiling code" OFF)
option(TENSOR_UNSAFE_SHARED_PTR "Non thread-safe version of shared_ptr" OFF)
set(TENSOR_BLAS "OpenBLAS" CACHE STRING "BLAS backend to use. Values are OpenBLAS (default) or MKL (also brings LAPACK backend)")

# Compiler settings and other tooling
make_tensor_options()
if(TENSOR_FORCE_STATIC)
  set(BUILD_SHARED_LIBS OFF)
endif()

# Search for dependencies
set(TENSOR_DEPENDENCIES)
set(cblas_found FALSE)
set(TENSOR_USE_LAPACK TRUE)
if(TENSOR_BLAS STREQUAL MKL)
  tensor_find_dependency(
    NAME MKL
    OUTVAR tensor_blas_target
    CONFIG_NAMES MKL
    IMPORT_TARGET MKL::MKL
    DEPENDENCIES TENSOR_DEPENDENCIES
    REQUIRED
  )
  set(cblas_found TRUE)
  set(TENSOR_USE_MKL TRUE)
  set(TENSOR_USE_LAPACK FALSE)
endif()
if((TENSOR_BLAS MATCHES OpenBLAS) OR (NOT cblas_found))
  tensor_find_dependency(
    NAME OpenBLAS
    OUTVAR tensor_blas_target
    CONFIG_NAMES OpenBLAS
    IMPORT_TARGET OpenBLAS::OpenBLAS
    DEPENDENCIES TENSOR_DEPENDENCIES
    PKGCONFIG_NAMES openblas-pthread openblas-openmp openblas openblas64
    REQUIRED
  )
  set(cblas_found TRUE)
  set(TENSOR_USE_OPENBLAS TRUE)
  set(HAVE_CBLAS_H TRUE)
  tensor_openblas_requires_lapack(TARGET ${tensor_blas_target}
    OUTVAR OPENBLAS_REQUIRES_LAPACK)
  set(TENSOR_USE_LAPACK ${OPENBLAS_REQUIRES_LAPACK})
else()
  message(FATAL_ERROR "TENSOR_BLAS must be one of MKL or OpenBLAS. Current value is \"${TENSOR_BLAS}\"")
endif()
if(TENSOR_USE_LAPACK)
  tensor_find_dependency(
    NAME lapack
    OUTVAR tensor_lapack_target
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
  set(TENSOR_USE_ARPACK TRUE)
  tensor_find_dependency(
    NAME arpack
    OUTVAR tensor_arpack_target
    CONFIG_NAMES arpackng arpack-ng ARPACK
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
    OUTVAR tensor_fftw_target
    CONFIG_NAMES FFTW3
    DEPENDENCIES TENSOR_DEPENDENCIES
    IMPORT_TARGET FFTW3 #::fftw3
    PKGCONFIG_NAMES fftw3
  )
  set(TENSOR_USE_FFTW3 TRUE)
endif()

if(TENSOR_PRIMME)
  set(TENSOR_USE_PRIMME TRUE)
  # We could simplify this if CMake version was 3.24 or later,
  # because it can default to find_package
  FetchContent_Declare(
	primme
	GIT_REPOSITORY https://github.com/juanjosegarciaripoll/primme.git
	GIT_TAG 5453190ca5ef9b13d1954feef2419cdd5d8c0deb
  )
  FetchContent_MakeAvailable(primme)
  set(TENSOR_DEPENDENCIES "${TENSOR_DEPENDENCIES}\nfind_dependency(primme)")
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
if(TENSOR_FFTW)
  target_link_libraries(tensor PRIVATE ${tensor_fftw_target})
endif()
if(TENSOR_ARPACK)
  target_link_libraries(tensor PUBLIC ${tensor_arpack_target})
endif()
if(TENSOR_PRIMME)
  target_link_libraries(tensor PUBLIC primme)
endif()
if(TENSOR_USE_LAPACK)
  target_link_libraries(tensor PRIVATE ${tensor_lapack_target})
endif()
target_link_libraries(tensor PUBLIC ${tensor_blas_target})
add_subdirectory(include)
add_subdirectory(src)

############################################################
# PROFILING
#
if(TENSOR_BUILD_PROFILE)
add_subdirectory("profile")
endif()

############################################################
# DOCUMENTATION
find_package(Doxygen)
if(DOXYGEN_FOUND)
  message(STATUS "Adding target for Doxygen documentation")
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_LATEX NO)
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_HAVE_DOT NO)
  set(DOXYGEN_SEARCHENGINE NO)
  set(DOXYGEN_SOURCE_BROWSER NO)
  set(DOXYGEN_VERBATIM_HEADERS NO)
  set(DOXYGEN_SHOW_USED_FILES NO)
  set(DOXYGEN_SHOW_FILES NO)
  set(DOXYGEN_CLASS_DIAGRAMS NO)
  set(DOXYGEN_STRIP_FROM_INC_PATH "include")
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_MATHJAX_VERSION "MathJax_3")
  set(DOXYGEN_MATHJAX_RELPATH "https://cdn.jsdelivr.net/npm/mathjax@3")
  set(DOXYGEN_MATHJAX_EXTENSIONS "ams")
  set(DOXYGEN_HTML_EXTRA_STYLESHEET "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen-awesome.css" "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen-awesome-sidebar-only.css")
  set(DOXYGEN_EXTRACT_LOCAL_CLASSES NO)
  set(DOXYGEN_EXCLUDE_PATTERNS "*/detail/*" "*/jobs.h" "*/arpack*.h" "*/tensor_blas.h" "*/tensor_lapack.h" "*/traits.h")
  doxygen_add_docs(
      doxygen
      ${CMAKE_CURRENT_SOURCE_DIR}/doc ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
      #ALL # Uncomment to make doxygen run on every build
      COMMENT "Generate documentation"
  )
endif()

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
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
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
