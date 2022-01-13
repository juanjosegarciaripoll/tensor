add_executable(profile prof_vector.cc)
target_link_libraries(profile PUBLIC tensor PRIVATE tensor_options)

if(TENSOR_BENCHMARK)
  add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/profile.json
    COMMAND profile ${PROJECT_BINARY_DIR}/profile.json
    COMMENT "Run benchmarks"
    VERBATIM
  )

  add_custom_target(
    ProfileCode ALL
    DEPENDS ${PROJECT_BINARY_DIR}/profile.json
  )
endif()