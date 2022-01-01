include_guard()

# On setup, try to find pkgconfig
find_package(PkgConfig QUIET)

function(tensor_find_dependency)
  # 0. Decipher arguments
  set(one_value_args NAME CONFIG_NAME IMPORT_TARGET)
  set(multi_value_args PKGCONFIG_NAMES)
  cmake_parse_arguments(ARGP
    ""
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT ARGP_NAME)
    message(FATAL_ERROR "tensor_find_dependency needs a NAME to find")
  endif()

  if (ARGP_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "tensor_find_dependency(): got invalid arguments '${ARGP_UNPARSED_ARGUMENTS}'")
  endif()

  # convenience definitions for later
  set(target_name Dependency::${ARGP_NAME})
  set(cxx_var_name "TENSOR_${ARGP_NAME}_CXXFLAGS")
  set(ld_var_name "TENSOR_${ARGP_NAME}_LDFLAGS")

  # 1. If there are variables for the target, create an interface target
  if (${cxx_var_name} OR ${ld_var_name})
    message(STATUS "Dependency '${ARGP_NAME}' set up from supplied variables.")
    message(STATUS "Values are CXX=${${cxx_var_name}}, LD=${${ld_var_name}}")
    add_library("${target_name}" INTERFACE)
    target_compile_options(${target_name} "${${cxx_var_name}}")
    target_link_options(${target_name} "${${ld_var_name}}")
    return()
  endif()

  # 2. Try to find the package with CMake find_package() and create a unified alias target
  if (ARGP_CONFIG_NAME AND ARGP_IMPORT_TARGET)
    find_package(${ARGP_CONFIG_NAME} CONFIG QUIET)
    if (${ARGP_CONFIG_NAME}_FOUND AND (TARGET "${ARGP_IMPORT_TARGET}"))
      message(STATUS "Dependency '${ARGP_NAME}' found with CMake")
      add_library(${target_name} ALIAS ${ARGP_IMPORT_TARGET})
      return()
    endif()
  endif()

  # 3. Try to find the dependency with pkgconfig
  if (PkgConfig_FOUND AND ARGP_PKGCONFIG_NAMES)
    pkg_search_module(${ARGP_NAME} REQUIRED QUIET
      IMPORTED_TARGET ${ARGP_PKGCONFIG_NAMES}
    )
    if (${ARGP_NAME}_FOUND)
      message(STATUS "Dependency '${ARGP_NAME}' found with pkgconfig")
      add_library(${target_name} ALIAS PkgConfig::${ARGP_NAME})
      return()
    endif()
  endif()

  message(FATAL_ERROR "Target '${ARGP_NAME}' not found")
endfunction()
