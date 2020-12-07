file(GLOB TENSOR_HEADER_FILES
  RELATIVE ${PROJECT_SOURCE_DIR}/include
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES false
  tensor/*.h
  tensor/*.hpp
  tensor/detail/*.h
  tensor/detail/*.hpp
  )

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/tensor
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  FILES
    ${CMAKE_BINARY_DIR}/include/tensor/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor
)
