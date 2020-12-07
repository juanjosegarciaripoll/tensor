add_subdirectory(tools)
add_subdirectory(rand)
add_subdirectory(indices)
add_subdirectory(sparse)
add_subdirectory(tensor)
add_subdirectory(compare)
add_subdirectory(generated)
add_subdirectory(linalg)
add_subdirectory(views)
add_subdirectory(sdf)

if(${TENSOR_FFTW})
  find_library(FFTW_LIBRARIES
    NAMES fftw3
    REQUIRED
    DOC "FFTW3 library")
  message(STATUS ${FFTW_LIBRARIES})
  target_link_libraries(tensor PUBLIC ${FFTW_LIBRARIES})
  add_subdirectory(fftw)
endif()

if(${TENSOR_ARPACK})
  add_subdirectory(arpack)
endif()
