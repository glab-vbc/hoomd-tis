#pragma once
#include "hoomd/ForceCompute.h"
#include <memory>
#include <pybind11/pybind11.h>
namespace hoomd { namespace md {
//! Minimal stub ForceCompute to verify the build pipeline (zeroes forces).
class PYBIND11_EXPORT TISStackingForceCompute : public ForceCompute {
public:
    TISStackingForceCompute(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~TISStackingForceCompute();
protected:
    virtual void computeForces(uint64_t timestep);
};
namespace detail { void export_TISStackingForceCompute(pybind11::module& m); }
}}
