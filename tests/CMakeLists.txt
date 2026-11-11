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
  unit/test_xgb.cpp
  unit/test_registration.cpp
  unit/test_dsl_expr.cpp
  unit/test_dsl_expand.cpp
  unit/test_dsl_lower.cpp
  unit/test_dsl_specs.cpp
  unit/test_index.cpp
  unit/test_cpu_backend.cpp
  unit/test_codegen.cpp)
target_link_libraries(blz_unit_tests PRIVATE blz_core blz_codegen Catch2::Catch2WithMain)
target_compile_definitions(blz_unit_tests PRIVATE
  BLZ_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  BLZ_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(blz_unit_tests PROPERTIES LABELS fast)

# SPEC §16 P3: the front end accepts the whole pattern library.
add_test(NAME blzc_check_library COMMAND blzc --check-only ${BLZ_PATTERN_SPECS})
set_tests_properties(blzc_check_library PROPERTIES LABELS fast)

# SPEC §13.4: the generated code equals tests/snapshots/. After reviewing a change, update with
#   (cd <source> && blzc patterns/*.yaml --snapshots tests/snapshots --update-snapshots)
add_test(NAME blzc_snapshots
  COMMAND blzc ${BLZ_PATTERN_SPECS_REL} --snapshots tests/snapshots
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(blzc_snapshots PROPERTIES LABELS fast)

# AC1 (SPEC §13.3): generated CPU vs oracle on 10,000 fuzz graphs, both schedules.
add_test(NAME fuzz_diff_cpu_10k COMMAND blz fuzz-diff --seeds 0:10000)
set_tests_properties(fuzz_diff_cpu_10k PROPERTIES LABELS slow)

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
