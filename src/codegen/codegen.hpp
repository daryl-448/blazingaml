#pragma once
#include <string>
#include <vector>

#include "ir/ir.hpp"

// Code generation (SPEC §7.1 steps 8–9, §7.4, §7.6): planned IR -> C++ sources.
namespace blz::codegen {

struct GeneratedFile {
  std::string name;  // file name inside the output directory
  std::string text;
};

// One pattern's sources, named after `stem` (its spec file's stem):
//   <stem>.hpp      <pattern>_trigger and <pattern>_work (BLZ_HD, shared with CUDA) and the
//                   declarations of <pattern>_cpu and the CUDA launchers;
//   <stem>_cpu.cpp  <pattern>_cpu, the OpenMP driver.
// Throws dsl::CompileError (code E501) for an IR the v1 emitter does not support.
std::vector<GeneratedFile> emit_cpu(const ir::Pattern& p, const std::string& stem);

// <stem>_cuda.cu (SPEC §7.5): the heavy kernel's body <pattern>_trigger_warp and the launchers
// <pattern>_cuda_bucket and <pattern>_cuda, which instantiate include/blz/cuda_kernels.cuh with
// the pattern's light body (<pattern>_trigger), heavy body and work estimate. Compiled only with
// BLZ_ENABLE_CUDA; always generated, so the snapshots do not depend on the build options.
GeneratedFile emit_cuda(const ir::Pattern& p, const std::string& stem);

namespace detail {
// The warp-mode lowering of cpu_emitter.cpp: <pattern>_trigger_warp's definition.
std::string warp_trigger_fn(const ir::Pattern& p);
}  // namespace detail

// registry.cpp: kPatterns[] for every pattern, in order. `stems` are the patterns' spec stems.
// Throws dsl::CompileError (E502) on two patterns with the same name.
GeneratedFile emit_registry(const std::vector<ir::Pattern>& patterns, const std::vector<std::string>& stems);

}  // namespace blz::codegen
