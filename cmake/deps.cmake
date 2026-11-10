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
# Used from P2 (xgboost) and P3 (yaml-cpp).
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG 0.8.0 GIT_SHALLOW TRUE)
FetchContent_Declare(xgboost
  GIT_REPOSITORY https://github.com/dmlc/xgboost.git GIT_TAG v3.4.2 GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(CLI11_BUILD_TESTS OFF CACHE INTERNAL "")
set(CLI11_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CLI11_BUILD_DOCS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json CLI11)
if(BUILD_TESTING)
  FetchContent_MakeAvailable(Catch2)
endif()
