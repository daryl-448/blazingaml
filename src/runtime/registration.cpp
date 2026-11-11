#include "runtime/registration.hpp"

#include <stdexcept>

namespace blz {

void check_registration(std::span<const PatternRef> patterns, const FeatureConfig& cfg,
                        const std::string& config_name) {
  for (const PatternRef& p : patterns) {
    const auto b = parse_block(p.block);
    if (!b)
      throw std::runtime_error(p.block_loc + ": error E402: pattern '" + p.name + "' emits into block '" +
                               p.block + "', which is not a feature block");
    if (!cfg[*b])
      throw std::runtime_error(p.block_loc + ": error E402: pattern '" + p.name + "' emits into block '" +
                               p.block + "', which the feature config " + config_name + " does not define");
  }
}

}  // namespace blz
