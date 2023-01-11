set(TENSOR_HEADER_FILES
  tensor/arpack.h
  tensor/detail/initializer.h
  tensor/detail/shared_ptr.h
  tensor/exceptions.h
  tensor/fftw.h
  tensor/flags.h
  tensor/indices.h
  tensor/io.h
  tensor/jobs.h
  tensor/linalg.h
  tensor/linalg/eigs.h
  tensor/linalg/operators.h
  tensor/numbers.h
  tensor/rand.h
  tensor/ranges.h
  tensor/sdf.h
  tensor/sparse/csr_matrix.h
  tensor/sparse/csr_operators.h
  tensor/sparse/types.h
  tensor/sparse.h
  tensor/tensor/accumulate.h
  tensor/tensor/ctensor.h
  tensor/tensor/implementation.h
  tensor/tensor/iterator.h
  tensor/tensor/mapping.h
  tensor/tensor/operators.h
  tensor/tensor/rtensor.h
  tensor/tensor/types.h
  tensor/tensor.h
  tensor/tensor_blas.h
  tensor/tensor_lapack.h
  tensor/tools.h
  tensor/traits.h
  tensor/vector.h
)

target_sources(tensor PRIVATE ${TENSOR_HEADER_FILES})

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/tensor
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  FILES
    ${CMAKE_BINARY_DIR}/include/tensor/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor
)
