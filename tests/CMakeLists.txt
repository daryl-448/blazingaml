include(Catch)

add_executable(blz_unit_tests
  unit/test_sha256.cpp
  unit/test_csv.cpp
  unit/test_prepare.cpp
  unit/test_blzg.cpp
  unit/test_npy.cpp
  unit/test_stats.cpp
  unit/test_features.cpp
  unit/test_fuzz_gen.cpp
  unit/test_oracle.cpp
  unit/test_metrics.cpp
  unit/test_xgb.cpp)
target_link_libraries(blz_unit_tests PRIVATE blz_core Catch2::Catch2WithMain)
target_compile_definitions(blz_unit_tests PRIVATE
  BLZ_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  BLZ_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(blz_unit_tests PROPERTIES LABELS fast)

# HI-Small checks (SPEC §13.5): run manually with `ctest -L slow`.
set(BLZ_HI_SMALL_CSV "${PROJECT_SOURCE_DIR}/dataset/HI-Small_Trans.csv")
add_test(NAME hi_small_prepare
  COMMAND blz prepare --csv ${BLZ_HI_SMALL_CSV} --out ${CMAKE_CURRENT_BINARY_DIR}/tmp/hi_small.blzg)
set_tests_properties(hi_small_prepare PROPERTIES LABELS slow)

if(BLZ_ENABLE_CUDA)
  add_executable(blz_cuda_smoke cuda/smoke.cu)
  add_test(NAME cuda_smoke COMMAND blz_cuda_smoke)
  set_tests_properties(cuda_smoke PROPERTIES LABELS fast)
endif()
