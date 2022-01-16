# TODO: Use FetchContent as in https://sourceforge.net/p/wavepacket/cpp/git/ci/master/tree/dependencies/CMakeLists.txt
# for sites where GTest might not be available, and to fix a certain version.

include(GoogleTest)
enable_testing()

mark_as_advanced(
  BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
  gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )

add_library(testmain
  test_main.cc alloc_informer.cc
  random_unitary_d.cc random_unitary_z.cc
  random_permutation.cc)
target_link_libraries(testmain
  PUBLIC tensor GTest::gtest)

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} testmain ${GTEST_PREFIX}gtest ${GTEST_PREFIX}gtest_main)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_booleans test_booleans.cc)
package_add_test(test_fold test_fold.cc)
package_add_test(test_index test_index.cc)
package_add_test(test_jobs test_jobs.cc)
package_add_test(test_linalg_cgs test_linalg_cgs.cc)
package_add_test(test_linalg_eig test_linalg_eig.cc)
package_add_test(test_linalg_eig_power test_linalg_eig_power.cc)
package_add_test(test_linalg_eig_sym test_linalg_eig_sym.cc)
package_add_test(test_linalg_expm test_linalg_expm.cc)
package_add_test(test_linalg_solve test_linalg_solve.cc)
package_add_test(test_linalg_svd test_linalg_svd.cc)
package_add_test(test_matrix test_matrix.cc)
package_add_test(test_mmult test_mmult.cc)
package_add_test(test_rand test_rand.cc)
target_include_directories(test_rand
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
#package_add_test(test_refcount test_refcount.cc)
package_add_test(test_sdf test_sdf.cc)
package_add_test(test_sparse_base test_sparse_base.cc)
package_add_test(test_sparse_indices test_sparse_indices.cc)
package_add_test(test_sparse_kron test_sparse_kron.cc)
package_add_test(test_sparse_ops test_sparse_ops.cc)
package_add_test(test_tensor_base test_tensor_base.cc)
package_add_test(test_tensor_binop test_tensor_binop.cc)
package_add_test(test_tensor_binop_error test_tensor_binop_error.cc)
package_add_test(test_tensor_comparison test_tensor_comparison.cc)
package_add_test(test_tensor_getset test_tensor_getset.cc)
package_add_test(test_tensor_kron test_tensor_kron.cc)
package_add_test(test_tensor_permute test_tensor_permute.cc)
package_add_test(test_tensor_scale test_tensor_scale.cc)
package_add_test(test_tensor_take_diag test_tensor_take_diag.cc)
package_add_test(test_tensor_trace test_tensor_trace.cc)
package_add_test(test_tensor_unop test_tensor_unop.cc)
package_add_test(test_vector test_vector.cc)
package_add_test(test_view_read test_view_read.cc)
package_add_test(test_view_errors test_view_errors.cc)
package_add_test(test_view_set test_view_set.cc)
if(${TENSOR_ARPACK})
  package_add_test(test_linalg_eigs test_linalg_eigs.cc)
  target_link_libraries(test_rand PRIVATE ${ARPACK_LIBRARY})
endif()
if(${TENSOR_FFTW})
  package_add_test(test_fftw test_fftw.cc)
endif()
