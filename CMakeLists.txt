cmake_minimum_required(VERSION 3.28)
project(blazingaml LANGUAGES CXX)

option(BLZ_ENABLE_CUDA "Build the CUDA backend" OFF)
option(BLZ_NATIVE "Compile with -march=native" ON)
option(BLZ_SANITIZE "Build with ASan + UBSan" OFF)
# blz train (SPEC §10). Off only where nothing trains, e.g. the temporary A30 (bench/a30.sh).
option(BLZ_WITH_XGBOOST "Build blz train against XGBoost (C API)" ON)
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
  find_package(CUDAToolkit REQUIRED)
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

# Code shared by the compiler and the runtime: SHA-256 (spec hashes in the registry, SPEC v2 §3.1;
# the CSV and .blzg hashes).
add_library(blz_common src/common/sha256.cpp)
target_include_directories(blz_common PUBLIC src)
target_link_libraries(blz_common PUBLIC blz_flags)

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
target_link_libraries(blz_dsl PUBLIC blz_flags PRIVATE blz_common yaml-cpp::yaml-cpp)

# Code generation (SPEC §7.1 steps 8-9): planned IR -> C++ sources.
add_library(blz_codegen
  src/codegen/cpu_emitter.cpp
  src/codegen/cuda_emitter.cpp
  src/codegen/registry_emitter.cpp)
target_link_libraries(blz_codegen PUBLIC blz_dsl)

add_executable(blzc src/apps/blzc.cpp)
target_link_libraries(blzc PRIVATE blz_codegen CLI11::CLI11)

# The generated pattern library (SPEC §7.1): blzc runs over patterns/*.yaml at build time and
# writes ${build}/generated/. It runs in the source directory with relative spec paths, so the
# generated text (file names in comments and E402 locations) equals tests/snapshots/. The CUDA
# files are always generated and compiled only with BLZ_ENABLE_CUDA.
file(GLOB BLZ_PATTERN_SPECS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/patterns/*.yaml)
list(SORT BLZ_PATTERN_SPECS)
set(BLZ_GEN_DIR ${PROJECT_BINARY_DIR}/generated)
set(BLZ_PATTERN_SPECS_REL)
set(BLZ_GEN_OUTPUTS ${BLZ_GEN_DIR}/registry.cpp)
set(BLZ_GEN_SOURCES ${BLZ_GEN_DIR}/registry.cpp)
foreach(spec IN LISTS BLZ_PATTERN_SPECS)
  file(RELATIVE_PATH rel ${PROJECT_SOURCE_DIR} ${spec})
  get_filename_component(stem ${spec} NAME_WLE)
  list(APPEND BLZ_PATTERN_SPECS_REL ${rel})
  list(APPEND BLZ_GEN_OUTPUTS ${BLZ_GEN_DIR}/${stem}.hpp ${BLZ_GEN_DIR}/${stem}_cpu.cpp ${BLZ_GEN_DIR}/${stem}_cuda.cu)
  list(APPEND BLZ_GEN_SOURCES ${BLZ_GEN_DIR}/${stem}_cpu.cpp)
  if(BLZ_ENABLE_CUDA)
    list(APPEND BLZ_GEN_SOURCES ${BLZ_GEN_DIR}/${stem}_cuda.cu)
  endif()
endforeach()
add_custom_command(OUTPUT ${BLZ_GEN_OUTPUTS}
  COMMAND blzc ${BLZ_PATTERN_SPECS_REL} --out ${BLZ_GEN_DIR}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  DEPENDS blzc ${BLZ_PATTERN_SPECS}
  COMMENT "blzc: generating the pattern library"
  VERBATIM)
add_library(blz_patterns ${BLZ_GEN_SOURCES})
target_include_directories(blz_patterns PUBLIC include ${BLZ_GEN_DIR})
target_link_libraries(blz_patterns PUBLIC blz_flags OpenMP::OpenMP_CXX)
if(BLZ_ENABLE_CUDA)
  # BLZ_WITH_CUDA: registry.cpp points at the CUDA launchers.
  target_compile_definitions(blz_patterns PUBLIC BLZ_WITH_CUDA=1)
  target_link_libraries(blz_patterns PUBLIC CUDA::cudart_static)
endif()

add_library(blz_core
  src/common/npy.cpp
  src/common/mmap_file.cpp
  src/io/csv.cpp
  src/io/blzg.cpp
  src/graph/prepare.cpp
  src/graph/stats.cpp
  src/graph/fuzz_gen.cpp
  src/graph/index.cpp
  src/graph/gt_join.cpp
  src/runtime/features.cpp
  src/runtime/profile.cpp
  src/runtime/triggers.cpp
  src/runtime/registration.cpp
  src/runtime/cpu_backend.cpp
  src/runtime/fuzz_diff.cpp
  src/runtime/metamorphic.cpp
  src/runtime/coverage.cpp
  src/oracle/oracle.cpp
  src/ml/metrics.cpp)
if(BLZ_ENABLE_CUDA)
  target_sources(blz_core PRIVATE src/runtime/cuda_backend.cu)
else()
  target_sources(blz_core PRIVATE src/runtime/cuda_backend_stub.cpp)
endif()
target_include_directories(blz_core PUBLIC include src)
target_link_libraries(blz_core PUBLIC blz_flags blz_common blz_patterns nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
  PRIVATE yaml-cpp::yaml-cpp)
if(BLZ_WITH_XGBOOST)
  target_sources(blz_core PRIVATE src/ml/xgb.cpp)
  target_link_libraries(blz_core PRIVATE xgboost)
  target_compile_definitions(blz_core PUBLIC BLZ_WITH_XGBOOST=1)
else()
  target_compile_definitions(blz_core PUBLIC BLZ_WITH_XGBOOST=0)
endif()

add_executable(blz src/apps/blz.cpp)
target_link_libraries(blz PRIVATE blz_core CLI11::CLI11)
# Recorded in benchmark results (SPEC §12.1).
string(TOUPPER "${CMAKE_BUILD_TYPE}" BLZ_BUILD_TYPE_UC)
set(BLZ_CXX_FLAGS_ALL "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BLZ_BUILD_TYPE_UC}}")
if(BLZ_NATIVE)
  string(APPEND BLZ_CXX_FLAGS_ALL " -march=native")
endif()
if(BLZ_SANITIZE)
  string(APPEND BLZ_CXX_FLAGS_ALL " -fsanitize=address,undefined")
endif()
string(STRIP "${BLZ_CXX_FLAGS_ALL}" BLZ_CXX_FLAGS_ALL)
if(BLZ_NATIVE)
  set(BLZ_NATIVE_FLAG true)
else()
  set(BLZ_NATIVE_FLAG false)
endif()
if(BLZ_ENABLE_CUDA)
  string(STRIP "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BLZ_BUILD_TYPE_UC}}" BLZ_CUDA_FLAGS_ALL)
  string(REPLACE ";" "," BLZ_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
  set(BLZ_CUDA_BUILD "nvcc ${CMAKE_CUDA_COMPILER_VERSION}, sm ${BLZ_CUDA_ARCHS}, ${BLZ_CUDA_FLAGS_ALL}")
else()
  set(BLZ_CUDA_BUILD "none")
endif()
target_compile_definitions(blz PRIVATE BLZ_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  BLZ_BUILD_TYPE="${CMAKE_BUILD_TYPE}" BLZ_CXX_FLAGS="${BLZ_CXX_FLAGS_ALL}" BLZ_NATIVE_FLAG=${BLZ_NATIVE_FLAG}
  BLZ_CUDA_BUILD="${BLZ_CUDA_BUILD}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
