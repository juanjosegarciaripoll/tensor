add_executable(profile prof_vector.cc)
target_link_libraries(profile PUBLIC tensor)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/profile_info.xml
  COMMAND profile ${PROJECT_BINARY_DIR}/profile_info.xml
  COMMENT "Run benchmarks"
  VERBATIM
)

add_custom_target(
  ProfileCode ALL
  DEPENDS ${PROJECT_BINARY_DIR}/profile_info.xml
)