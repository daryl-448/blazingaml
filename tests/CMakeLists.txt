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
  unit/test_registration.cpp
  unit/test_dsl_expr.cpp
  unit/test_dsl_expand.cpp
  unit/test_dsl_lower.cpp
  unit/test_dsl_specs.cpp
  unit/test_index.cpp
  unit/test_cpu_backend.cpp
  unit/test_codegen.cpp
  unit/test_profile.cpp
  unit/test_triggers.cpp
  unit/test_metamorphic.cpp
  unit/test_gt_join.cpp)
if(BLZ_WITH_XGBOOST)
  target_sources(blz_unit_tests PRIVATE unit/test_xgb.cpp)
endif()
if(BLZ_ENABLE_CUDA)
  target_sources(blz_unit_tests PRIVATE unit/test_cuda_backend.cpp)
endif()
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
if(BLZ_ENABLE_CUDA)
  # The same for CUDA, with every trigger on the heavy kernel, then mixed, then on the light one.
  add_test(NAME fuzz_diff_cuda_10k COMMAND blz fuzz-diff --backend cuda --seeds 0:10000)
  set_tests_properties(fuzz_diff_cuda_10k PROPERTIES LABELS slow)
endif()

# SPEC v2 §3.6: the DG-2 ledger's budget, stop rule and entry checks.
add_test(NAME dg2_ledger_selftest COMMAND python3 ${PROJECT_SOURCE_DIR}/bench/dg2/ledger.py selftest)
set_tests_properties(dg2_ledger_selftest PROPERTIES LABELS fast)

# SPEC v2 §3.4, §10: the metamorphic tests on 2,000 fuzz graphs per frozen profile and backend.
foreach(prof v1_default v1_parity)
  foreach(be oracle cpu)
    add_test(NAME metamorphic_${prof}_${be}_2k
      COMMAND blz metamorphic --seeds 0:2000 --backend ${be} --profile ${PROJECT_SOURCE_DIR}/configs/profiles/${prof}.yaml)
    set_tests_properties(metamorphic_${prof}_${be}_2k PROPERTIES LABELS slow)
  endforeach()
  if(BLZ_ENABLE_CUDA)
    add_test(NAME metamorphic_${prof}_cuda_2k
      COMMAND blz metamorphic --seeds 0:2000 --backend cuda --profile ${PROJECT_SOURCE_DIR}/configs/profiles/${prof}.yaml)
    set_tests_properties(metamorphic_${prof}_cuda_2k PROPERTIES LABELS slow)
  endif()
endforeach()

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
