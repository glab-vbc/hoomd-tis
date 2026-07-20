#include "TISForces.h"
#include "MixedPrecisionCompat.h"
#include <cstring>
namespace hoomd { namespace md {
TISStackingForceCompute::TISStackingForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef) {}
TISStackingForceCompute::~TISStackingForceCompute() {}
void TISStackingForceCompute::computeForces(uint64_t timestep) {
    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);
    std::memset((void*)h_force.data, 0, sizeof(ForceReal4) * m_force.getNumElements());
}
namespace detail {
void export_TISStackingForceCompute(pybind11::module& m) {
    pybind11::class_<TISStackingForceCompute, ForceCompute,
        std::shared_ptr<TISStackingForceCompute>>(m, "TISStackingForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
}
}}}
