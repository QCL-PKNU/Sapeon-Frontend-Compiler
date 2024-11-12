#ifndef ENUMS_TO_UNDERLYING_TYPE_HPP
#define ENUMS_TO_UNDERLYING_TYPE_HPP

#include <type_traits>

namespace spgraph_simulator {

template <typename ScopedEnum>
constexpr auto ToUnderlyingType(ScopedEnum e) noexcept {
  return static_cast<std::underlying_type_t<ScopedEnum>>(e);
}

}  // namespace spgraph_simulator

#endif  // ENUMS_TO_UNDERLYING_TYPE_HPP
