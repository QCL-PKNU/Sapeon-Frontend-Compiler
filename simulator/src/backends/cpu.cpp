#include "backends/cpu.hpp"

#define BASE Backend
#define NAME cpu
#define CLASS CpuBackend
#define OPERATION_CLASS CpuOperation
#define SCOPE CLASS
#define STR(x) #x
#define GET_STR(x) STR(x)

#include <memory>

#include "backends/backend.hpp"
#include "factory.hpp"

static bool kRegistered =
    Factory<BASE>::RegisterCreateFunction(GET_STR(NAME), CLASS::CreateBackend);

std::unique_ptr<BASE> SCOPE::CreateBackend() {
  return std::make_unique<CLASS>();
}
