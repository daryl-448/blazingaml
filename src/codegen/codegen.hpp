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
//   <stem>.hpp      <pattern>_trigger and <pattern>_work (BLZ_HD, shared with CUDA in P5) and
//                   the declaration of <pattern>_cpu;
//   <stem>_cpu.cpp  <pattern>_cpu, the OpenMP driver.
// Throws dsl::CompileError (code E501) for an IR the v1 emitter does not support.
std::vector<GeneratedFile> emit_cpu(const ir::Pattern& p, const std::string& stem);

// registry.cpp: kPatterns[] for every pattern, in order. `stems` are the patterns' spec stems.
// Throws dsl::CompileError (E502) on two patterns with the same name.
GeneratedFile emit_registry(const std::vector<ir::Pattern>& patterns, const std::vector<std::string>& stems);

}  // namespace blz::codegen
