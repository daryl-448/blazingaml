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
  unit/test_clean_oracle.cpp
  unit/test_vstats.cpp
  unit/test_lcyc.cpp
  unit/test_set_ops.cpp
  unit/test_typologies.cpp
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
  unit/test_gt_join.cpp
  unit/test_coverage.cpp
  unit/test_sweep.cpp
  unit/test_reach.cpp
  unit/test_fusion.cpp
  unit/test_clean.cpp)
if(BLZ_WITH_XGBOOST)
  target_sources(blz_unit_tests PRIVATE unit/test_xgb.cpp)
endif()
if(BLZ_ENABLE_CUDA)
  target_sources(blz_unit_tests PRIVATE unit/test_cuda_backend.cpp unit/test_hybrid.cpp unit/test_pack_fuse.cpp
    unit/test_clean_cuda.cpp)
endif()
target_link_libraries(blz_unit_tests PRIVATE blz_core blz_codegen Catch2::Catch2WithMain)
# The vertex-statistics references must round as vstats.cpp and clean_oracle.cpp do (SPEC v2 §8 S2, v3 R8).
set_source_files_properties(unit/test_vstats.cpp unit/test_clean_oracle.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
target_compile_definitions(blz_unit_tests PRIVATE
  BLZ_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  BLZ_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(blz_unit_tests PROPERTIES LABELS fast)

# SPEC §16 P3: the front end accepts the whole pattern library.
add_test(NAME blzc_check_library COMMAND blzc --check-only ${BLZ_PATTERN_SPECS})
set_tests_properties(blzc_check_library PROPERTIES LABELS fast)

# SPEC §13.4: the generated code equals tests/snapshots/, the fusions included (SPEC v2 §8 S7).
# After reviewing a change, update with
#   (cd <source> && blzc --fuse fan_in,deg_in --fuse fan_out,deg_out --fuse clean_fan_in,clean_deg_in
#      --fuse clean_fan_out,clean_deg_out patterns/*.yaml patterns/clean/*.yaml
#      --snapshots tests/snapshots --update-snapshots)
add_test(NAME blzc_snapshots
  COMMAND blzc ${BLZ_FUSE_ARGS} ${BLZ_PATTERN_SPECS_REL} --snapshots tests/snapshots
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(blzc_snapshots PROPERTIES LABELS fast)

# SPEC v2 §10: each blzc --disable variant (§7.2) of a representative spec: sg_gather has scans,
# distinct scans and a count-only intersection; fan_out is the sweep's (§7.1). Update with
#   (cd <source> && blzc <spec> --disable <opt> --snapshots tests/snapshots_disable/<opt> --update-snapshots)
foreach(opt find_start distinct_first degree_order count_only sweep)
  set(spec patterns/sg_gather.yaml)
  if(opt STREQUAL "sweep")
    set(spec patterns/fan_out.yaml)
  endif()
  add_test(NAME blzc_snapshots_disable_${opt}
    COMMAND blzc ${spec} --disable ${opt} --snapshots tests/snapshots_disable/${opt}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(blzc_snapshots_disable_${opt} PROPERTIES LABELS fast)
endforeach()

# AC1 (SPEC §13.3): generated CPU vs oracle on 10,000 fuzz graphs, both schedules.
add_test(NAME fuzz_diff_cpu_10k COMMAND blz fuzz-diff --seeds 0:10000)
set_tests_properties(fuzz_diff_cpu_10k PROPERTIES LABELS slow)
if(BLZ_ENABLE_CUDA)
  # The same for CUDA, with every trigger on the heavy kernel, then mixed, then on the light one.
  add_test(NAME fuzz_diff_cuda_10k COMMAND blz fuzz-diff --backend cuda --seeds 0:10000)
  set_tests_properties(fuzz_diff_cuda_10k PROPERTIES LABELS slow)
  # SPEC v2 §8 S6: the hybrid (heavy triggers on the CPU, pruned, the light ones on the GPU), every
  # pair of GPU thresholds 0, 4, all and CPU thresholds 0, 4, all, under both visibilities.
  add_test(NAME fuzz_diff_hybrid_10k COMMAND blz fuzz-diff --backend hybrid --seeds 0:10000)
  add_test(NAME fuzz_diff_hybrid_tsvis_10k COMMAND blz fuzz-diff --backend hybrid --seeds 0:10000
    --profile ${PROJECT_SOURCE_DIR}/configs/profiles/sg_ordered_tsvis.yaml)
  # Nine runs per graph on tiny graphs: two OpenMP threads spin less than twelve (about 8 min each).
  set_tests_properties(fuzz_diff_hybrid_10k fuzz_diff_hybrid_tsvis_10k PROPERTIES LABELS slow
    ENVIRONMENT OMP_NUM_THREADS=2 TIMEOUT 3000)
endif()

if(BLZ_ENABLE_CUDA)
  # SPEC v2 §8 S7: fused kernels and packed D2H, H = 0, 4 and all, under both visibilities.
  add_test(NAME fuzz_diff_cuda_s7_10k COMMAND blz fuzz-diff --backend cuda --fuse --pack --seeds 0:10000)
  add_test(NAME fuzz_diff_cuda_s7_tsvis_10k COMMAND blz fuzz-diff --backend cuda --fuse --pack --seeds 0:10000
    --profile ${PROJECT_SOURCE_DIR}/configs/profiles/sg_ordered_tsvis.yaml)
  set_tests_properties(fuzz_diff_cuda_s7_10k fuzz_diff_cuda_s7_tsvis_10k PROPERTIES LABELS slow)
endif()

# SPEC v2 §3.6: the DG-2 ledger's budget, stop rule and entry checks.
add_test(NAME dg2_ledger_selftest COMMAND python3 ${PROJECT_SOURCE_DIR}/bench/dg2/ledger.py selftest)
set_tests_properties(dg2_ledger_selftest PROPERTIES LABELS fast)

# SPEC v2 §3.4, §10: the metamorphic tests on 2,000 fuzz graphs per frozen profile and backend.
foreach(prof v1_default v1_parity sg_ordered_v1w sg_ordered_tsvis)
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

# AC-V2-4 (SPEC v2 §6, P10): AC1 per sg_ordered profile, i.e. sg_ord under both visibilities and
# the six v1 blocks under timestamp visibility. CUDA runs H = 0, 4 and all.
foreach(prof sg_ordered_v1w sg_ordered_tsvis)
  add_test(NAME fuzz_diff_${prof}_cpu_10k
    COMMAND blz fuzz-diff --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/${prof}.yaml)
  set_tests_properties(fuzz_diff_${prof}_cpu_10k PROPERTIES LABELS slow)
  if(BLZ_ENABLE_CUDA)
    add_test(NAME fuzz_diff_${prof}_cuda_10k
      COMMAND blz fuzz-diff --backend cuda --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/${prof}.yaml)
    set_tests_properties(fuzz_diff_${prof}_cuda_10k PROPERTIES LABELS slow)
  endif()
endforeach()

# SPEC v2 §8 S3: AC1 for lcyc (length <= 10) on 10,000 fuzz graphs. A graph whose oracle run
# reaches the work limit is reported and its stop eid compared, not its rows (D-39): 10^6 on the
# CPU and CUDA's H = 0 and 4; 10^5 for H = all, where one GPU thread walks each trigger's bindings.
add_test(NAME fuzz_diff_lcyc_f2_cpu_10k
  COMMAND blz fuzz-diff --seeds 0:10000 --work-limit 1000000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/lcyc_f2.yaml)
set_tests_properties(fuzz_diff_lcyc_f2_cpu_10k PROPERTIES LABELS slow)
if(BLZ_ENABLE_CUDA)
  add_test(NAME fuzz_diff_lcyc_f2_cuda_10k
    COMMAND blz fuzz-diff --backend cuda --heavy 0,4 --seeds 0:10000 --work-limit 1000000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/lcyc_f2.yaml)
  set_tests_properties(fuzz_diff_lcyc_f2_cuda_10k PROPERTIES LABELS slow)
  add_test(NAME fuzz_diff_lcyc_f2_cuda_light_10k
    COMMAND blz fuzz-diff --backend cuda --heavy all --seeds 0:10000 --work-limit 100000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/lcyc_f2.yaml)
  set_tests_properties(fuzz_diff_lcyc_f2_cuda_light_10k PROPERTIES LABELS slow)
endif()

# SPEC v2 §8 S4: AC1 for the set ops' features, fan_oneway_out (difference) and fan_total (union),
# on 10,000 fuzz graphs. CUDA runs H = 0, 4 and all.
add_test(NAME fuzz_diff_fan_sets_cpu_10k
  COMMAND blz fuzz-diff --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/fan_sets.yaml)
set_tests_properties(fuzz_diff_fan_sets_cpu_10k PROPERTIES LABELS slow)
if(BLZ_ENABLE_CUDA)
  add_test(NAME fuzz_diff_fan_sets_cuda_10k
    COMMAND blz fuzz-diff --backend cuda --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/fan_sets.yaml)
  set_tests_properties(fuzz_diff_fan_sets_cuda_10k PROPERTIES LABELS slow)
endif()

# SPEC v2 §8 S5: AC1 for the typologies' features, tchain (temporal chains, witness: latest) and
# gs (gather-scatter, min()), on 10,000 fuzz graphs. CUDA runs H = 0, 4 and all.
add_test(NAME fuzz_diff_typologies_cpu_10k
  COMMAND blz fuzz-diff --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/typologies.yaml)
set_tests_properties(fuzz_diff_typologies_cpu_10k PROPERTIES LABELS slow)
if(BLZ_ENABLE_CUDA)
  add_test(NAME fuzz_diff_typologies_cuda_10k
    COMMAND blz fuzz-diff --backend cuda --seeds 0:10000 --profile ${PROJECT_SOURCE_DIR}/configs/profiles/typologies.yaml)
  set_tests_properties(fuzz_diff_typologies_cuda_10k PROPERTIES LABELS slow)
endif()

# SPEC v2 §6.1 (P8): the oracle under timestamp visibility, including tcyc's visibility invariance.
add_test(NAME metamorphic_tsvis_oracle_2k
  COMMAND blz metamorphic --seeds 0:2000 --backend oracle --profile ${CMAKE_CURRENT_SOURCE_DIR}/profiles/tsvis.yaml)
set_tests_properties(metamorphic_tsvis_oracle_2k PROPERTIES LABELS slow)

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
