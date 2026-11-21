#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "blz/registry.hpp"
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

// A fusion (blzc --fuse, SPEC v2 §8 S7): sweep-shaped patterns over the same trigger list, which one
// CUDA kernel body counts with one scan (D-46). Each member writes its own block's rows; the
// members' distinct blocks are the slots, numbered in the order the members first name them.
struct Fusion {
  std::string id;                       // the members' names joined by "__": file and function names
  std::string name;                     // the members' names joined by "+"
  std::vector<std::size_t> members;     // indices into the pattern list, in --fuse order
  std::vector<std::uint32_t> slots;     // per member: its block's slot
  std::vector<std::string> first_scans; // per member: the trigger role and direction of its first scan ("dst.in")
};
// The fusions named by `groups` (pattern names). Throws std::runtime_error on a group of fewer than
// 2 or more than kMaxFusedMembers patterns, an unknown name, a pattern in two groups, a member with
// `work_limit: true` (the fused kernels do not count bindings), two members that pin different
// visibilities, a member without the sweep shape (Plan::sweep: its only stage is a root count over
// the trigger's src.out or dst.in, bounded by the window alone), or members that scan different lists.
std::vector<Fusion> plan_fusions(const std::vector<ir::Pattern>& patterns,
                                 const std::vector<std::vector<std::string>>& groups);
// fused_<id>_cuda.cu: the fusion's body (fused_<id>_kernels, one shared scan) and its launchers
// fused_<id>_cuda_bucket and fused_<id>_cuda, which instantiate include/blz/cuda_kernels.cuh's
// fused kernels.
GeneratedFile emit_fused_cuda(const Fusion& f, const std::vector<ir::Pattern>& patterns);

namespace detail {
// The warp-mode lowering of cpu_emitter.cpp: <pattern>_trigger_warp's definition.
std::string warp_trigger_fn(const ir::Pattern& p);
// Scan fusion (SPEC v2 §8 S7): <name>_trigger(g, p, e, lane, stride, sinks), one scan of the shared
// trigger list counting for every member, and <name>_work(g, p, e). Every member must have the
// sweep shape (Plan::sweep) over the same trigger role and direction; member k emits into
// sinks[slots[k]].
std::string scan_fusion_fns(const std::string& name, const std::vector<const ir::Pattern*>& members,
                            const std::vector<std::uint32_t>& slots);
}  // namespace detail

// registry.cpp: kPatterns[] for every pattern, in order, and kFusions[] for `fusions`. `stems` are
// the patterns' spec stems. Throws dsl::CompileError (E102) on two patterns with the same name.
GeneratedFile emit_registry(const std::vector<ir::Pattern>& patterns, const std::vector<std::string>& stems,
                            const std::vector<Fusion>& fusions = {});

}  // namespace blz::codegen
