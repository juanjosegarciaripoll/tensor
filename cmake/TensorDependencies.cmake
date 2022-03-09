include_guard()

# On setup, try to find pkgconfig
find_package(PkgConfig QUIET)

function(tensor_find_dependency)
  # 0. Decipher arguments
  set(option_args REQUIRED)
  set(one_value_args NAME VAR IMPORT_TARGET DEPENDENCIES)
  set(multi_value_args CONFIG_NAMES PKGCONFIG_NAMES)
  cmake_parse_arguments(ARGP
    "${option_args}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  if (NOT ARGP_DEPENDENCIES)
    message(FATAL_ERROR "ARGP_DEPENDENCIES variable is required")
  endif() 

  if (NOT ARGP_NAME)
    message(FATAL_ERROR "tensor_find_dependency needs a NAME to find")
  endif()

  if (NOT ARGP_VAR)
    message(FATAL_ERROR "tensor_find_dependency needs a VAR to output the result")
  endif()

  if (ARGP_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "tensor_find_dependency(): got invalid arguments '${ARGP_UNPARSED_ARGUMENTS}'")
  endif()

  # convenience definitions for later
  set(cxx_var_name "TENSOR_${ARGP_NAME}_CXXFLAGS")
  set(ld_var_name "TENSOR_${ARGP_NAME}_LDFLAGS")

  # 1. If there are variables for the target, create an interface target
  if (${cxx_var_name} OR ${ld_var_name})
    set(target_name Dependency::${ARGP_NAME})
    message(STATUS "Dependency '${ARGP_NAME}' set up from supplied variables.")
    add_library("${target_name}" INTERFACE)
    set("${ARGP_VAR}" "${target_name}" PARENT_SCOPE)
    target_compile_options(${target_name} "${${cxx_var_name}}")
    target_link_options(${target_name} "${${ld_var_name}}")
    set("${ARGP_NAME}_FOUND" TRUE PARENT_SCOPE)
    return()
  endif()

  # 2. Try to find the package with CMake find_package() and create a unified alias target
  if (ARGP_CONFIG_NAMES AND ARGP_IMPORT_TARGET)
    foreach(ARGP_CONFIG_NAME ${ARGP_CONFIG_NAMES})
      find_package(${ARGP_CONFIG_NAME} CONFIG QUIET)
      if (${ARGP_CONFIG_NAME}_FOUND AND (TARGET "${ARGP_IMPORT_TARGET}"))
        message(STATUS "Dependency '${ARGP_NAME}' found with CMake")
        set("${ARGP_VAR}" "${ARGP_IMPORT_TARGET}" PARENT_SCOPE)
        set("${ARGP_DEPENDENCIES}" "${${ARGP_DEPENDENCIES}}\nfind_dependency(${ARGP_CONFIG_NAME})" PARENT_SCOPE)
        set("${ARGP_NAME}_FOUND" TRUE PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()

  # 3. Try to find the dependency with pkgconfig
  if (ARGP_PKGCONFIG_NAMES)
    if (PkgConfig_FOUND)
      pkg_search_module(${ARGP_NAME} REQUIRED QUIET
        IMPORTED_TARGET ${ARGP_PKGCONFIG_NAMES}
      )
      if (${ARGP_NAME}_FOUND)
        message(STATUS "Dependency '${ARGP_NAME}' found with pkgconfig")
        set("${ARGP_VAR}" "PkgConfig::${ARGP_NAME}" PARENT_SCOPE)
        set("${ARGP_DEPENDENCIES}" "${${ARGP_DEPENDENCIES}}\nfind_dependency(PkgConfig)\npkg_search_module(${ARGP_NAME} REQUIRED QUIET IMPORTED_TARGET ${ARGP_PKGCONFIG_NAMES})" PARENT_SCOPE)
        set("${ARGP_NAME}_FOUND" TRUE PARENT_SCOPE)
        return()
      endif()
    else()
      message(STATUS "PkgConfig is not installed. Cannot search for ${ARGP_PKGCONFIG_NAMES}")
    endif()
  endif()

  if (ARGP_REQUIRED)
    message(FATAL_ERROR "Target '${ARGP_NAME}' not found")
  endif()
endfunction()
