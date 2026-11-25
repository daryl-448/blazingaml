# Third-party dependencies, each pinned to an exact tag, or a commit where there is none (SPEC §2.2).
# Only the ones a phase needs are made available; the rest are declared here so the pins live
# in one place.
include(FetchContent)

FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.16.1 GIT_SHALLOW TRUE)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0 GIT_SHALLOW TRUE)
FetchContent_Declare(CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.7.2 GIT_SHALLOW TRUE)
# yaml-cpp: feature configs (P1) and specs (P3); 0.8.0 lacks <cstdint> includes for GCC 16.
# xgboost: the classifier (P2), C API, static library.
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG yaml-cpp-0.9.0 GIT_SHALLOW TRUE)
FetchContent_Declare(xgboost
  GIT_REPOSITORY https://github.com/dmlc/xgboost.git GIT_TAG v3.4.2 GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE)
# RapidCheck: L3's properties (SPEC v3 §4.3), test builds only. It has no tags, so the pin is a
# commit. The library alone, without its submodules (an old Catch, googletest): RC_ENABLE_CATCH
# would build that old Catch, and its rc::prop takes one seed and case count per process
# (RC_PARAMS). The properties run in Catch2 v3 test cases through check() in
# tests/unit/test_l3.cpp, which gives each its own seed and case count.
FetchContent_Declare(rapidcheck
  GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git GIT_TAG 2c3c4365aca21ef4e612768fdc95b1ce8b39a651
  GIT_SUBMODULES "")

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(CLI11_BUILD_TESTS OFF CACHE INTERNAL "")
set(CLI11_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CLI11_BUILD_DOCS OFF CACHE INTERNAL "")
set(YAML_CPP_BUILD_TESTS OFF CACHE INTERNAL "")
set(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE INTERNAL "")
set(YAML_CPP_INSTALL OFF CACHE INTERNAL "")
set(BUILD_STATIC_LIB ON CACHE INTERNAL "")
set(USE_OPENMP ON CACHE INTERNAL "")
set(USE_CUDA OFF CACHE INTERNAL "")  # CPU-only XGBoost (D-13)
set(RC_ENABLE_TESTS OFF CACHE INTERNAL "")
set(RC_ENABLE_EXAMPLES OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json CLI11 yaml-cpp)
if(BLZ_WITH_XGBOOST)
  FetchContent_MakeAvailable(xgboost)
endif()
if(BUILD_TESTING)
  FetchContent_MakeAvailable(Catch2 rapidcheck)
endif()
