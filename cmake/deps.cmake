# Third-party dependencies, each pinned to an exact tag (SPEC §2.2).
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
FetchContent_MakeAvailable(nlohmann_json CLI11 yaml-cpp xgboost)
if(BUILD_TESTING)
  FetchContent_MakeAvailable(Catch2)
endif()
