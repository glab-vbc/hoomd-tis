#include <pybind11/pybind11.h>
namespace hoomd { namespace md { namespace detail {
void export_TISStackingForceCompute(pybind11::module& m);
}}}
using namespace hoomd::md::detail;
PYBIND11_MODULE(_engine, m) { export_TISStackingForceCompute(m); }
