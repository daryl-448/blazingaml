cmake_minimum_required(VERSION 3.28)
project(blazingaml LANGUAGES CXX)

option(BLZ_ENABLE_CUDA "Build the CUDA backend" OFF)
option(BLZ_NATIVE "Compile with -march=native" ON)
option(BLZ_SANITIZE "Build with ASan + UBSan" OFF)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BLZ_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
include(cmake/deps.cmake)

add_library(blz_flags INTERFACE)
target_compile_options(blz_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${BLZ_NATIVE}>>:-march=native>)
if(BLZ_SANITIZE)
  target_compile_options(blz_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-omit-frame-pointer>)
  target_link_options(blz_flags INTERFACE -fsanitize=address,undefined)
endif()

add_library(blz_core
  src/common/sha256.cpp
  src/common/npy.cpp
  src/common/mmap_file.cpp
  src/io/csv.cpp
  src/io/blzg.cpp
  src/graph/prepare.cpp
  src/graph/stats.cpp
  src/graph/fuzz_gen.cpp
  src/runtime/features.cpp
  src/runtime/registration.cpp
  src/oracle/oracle.cpp
  src/ml/metrics.cpp
  src/ml/xgb.cpp)
target_include_directories(blz_core PUBLIC include src)
target_link_libraries(blz_core PUBLIC blz_flags nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
  PRIVATE yaml-cpp::yaml-cpp xgboost)

# The pattern compiler's front end (SPEC §7.1): YAML -> AST -> planned IR. A host tool library;
# it does not depend on the runtime.
add_library(blz_dsl
  src/dsl/ynode.cpp
  src/dsl/expand.cpp
  src/dsl/expr.cpp
  src/dsl/parse.cpp
  src/ir/lower.cpp
  src/ir/plan.cpp
  src/ir/dump.cpp
  src/ir/compile.cpp)
target_include_directories(blz_dsl PUBLIC src)
target_link_libraries(blz_dsl PUBLIC blz_flags PRIVATE yaml-cpp::yaml-cpp)

add_executable(blzc src/apps/blzc.cpp)
target_link_libraries(blzc PRIVATE blz_dsl CLI11::CLI11)

add_executable(blz src/apps/blz.cpp)
target_link_libraries(blz PRIVATE blz_core CLI11::CLI11)
target_compile_definitions(blz PRIVATE BLZ_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
