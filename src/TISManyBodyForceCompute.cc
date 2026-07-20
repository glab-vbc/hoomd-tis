// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file TISManyBodyForceCompute.cc
    \brief CPU implementation of the generic TIS many-body force.

    Ports tis/custom_forces.py (the validated reference) verbatim: minimum-image
    edge vectors, per-coordinate value + Cartesian gradient (dist / angle /
    Blondel-Karplus dihedral), the rational energy U = U0 / D with
    D = 1 + Sum_a k_a dq_a^2, and the analytic chain-rule force. All internal
    math is done in double (Scalar == double on this mixed-precision fork); only
    the final force / per-particle energy written to m_force is single precision
    (ForceReal4).
*/

#include "TISForces.h"
#include "MixedPrecisionCompat.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
namespace md
    {

// --------------------------------------------------------------------------- //
// Small double3 vector helpers (Scalar == double on this build)               //
// --------------------------------------------------------------------------- //
namespace
    {

inline Scalar3 v_sub(const Scalar3& a, const Scalar3& b)
    {
    return make_scalar3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
inline Scalar3 v_scale(const Scalar3& a, double s)
    {
    return make_scalar3(a.x * s, a.y * s, a.z * s);
    }
inline double v_dot(const Scalar3& a, const Scalar3& b)
    {
    return a.x * b.x + a.y * b.y + a.z * b.z;
    }
inline Scalar3 v_cross(const Scalar3& a, const Scalar3& b)
    {
    return make_scalar3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
inline double v_norm(const Scalar3& a)
    {
    return sqrt(v_dot(a, a));
    }

//! Wrap an angular difference into (-pi, pi] (identity derivative).
inline double wrap(double delta)
    {
    return atan2(sin(delta), cos(delta));
    }

//! Result of evaluating one coordinate: its value and per-particle gradients.
struct CoordEval
    {
    double q;
    int n;
    Scalar3 g[4];
    };

//! q = |pi - pj|; grads [u, -u], u = d/q.
inline void eval_dist(const Scalar3& pi, const Scalar3& pj, const BoxDim& box, CoordEval& e)
    {
    Scalar3 d = box.minImage(v_sub(pi, pj));
    double q = v_norm(d);
    Scalar3 u = v_scale(d, 1.0 / q);
    e.q = q;
    e.n = 2;
    e.g[0] = u;
    e.g[1] = v_scale(u, -1.0);
    }

//! Bond angle at the middle particle p1, between (p0-p1) and (p2-p1).
inline void
eval_angle(const Scalar3& p0, const Scalar3& p1, const Scalar3& p2, const BoxDim& box, CoordEval& e)
    {
    Scalar3 v1 = box.minImage(v_sub(p0, p1));
    Scalar3 v2 = box.minImage(v_sub(p2, p1));
    double r1 = v_norm(v1);
    double r2 = v_norm(v2);
    Scalar3 u1 = v_scale(v1, 1.0 / r1);
    Scalar3 u2 = v_scale(v2, 1.0 / r2);
    double c = v_dot(u1, u2);
    if (c > 1.0)
        c = 1.0;
    if (c < -1.0)
        c = -1.0;
    double s = sqrt(fmax(1.0 - c * c, 1e-12)); // guard near 0 / 180 deg
    e.q = acos(c);
    e.n = 3;
    // g0 = -(u2 - c u1)/(r1 s)
    e.g[0] = v_scale(v_sub(u2, v_scale(u1, c)), -1.0 / (r1 * s));
    // g2 = -(u1 - c u2)/(r2 s)
    e.g[2] = v_scale(v_sub(u1, v_scale(u2, c)), -1.0 / (r2 * s));
    // g1 = -(g0 + g2)
    e.g[1] = v_scale(make_scalar3(e.g[0].x + e.g[2].x, e.g[0].y + e.g[2].y, e.g[0].z + e.g[2].z),
                     -1.0);
    }

//! Dihedral phi about the p1-p2 axis (Blondel-Karplus gradient).
inline void eval_dihedral(const Scalar3& p0,
                          const Scalar3& p1,
                          const Scalar3& p2,
                          const Scalar3& p3,
                          const BoxDim& box,
                          CoordEval& e)
    {
    Scalar3 b1 = box.minImage(v_sub(p1, p0));
    Scalar3 b2 = box.minImage(v_sub(p2, p1));
    Scalar3 b3 = box.minImage(v_sub(p3, p2));
    Scalar3 n1 = v_cross(b1, b2);
    Scalar3 n2 = v_cross(b2, b3);
    double b2n = v_norm(b2);
    double n1sq = v_dot(n1, n1);
    double n2sq = v_dot(n2, n2);
    Scalar3 m1 = v_cross(n1, v_scale(b2, 1.0 / b2n));
    e.q = atan2(v_dot(m1, n2), v_dot(n1, n2));
    e.n = 4;

    Scalar3 g0 = v_scale(n1, b2n / n1sq);
    Scalar3 g3 = v_scale(n2, -b2n / n2sq);
    double a = v_dot(b1, b2) / (b2n * b2n);
    double c = v_dot(b3, b2) / (b2n * b2n);
    // g1 = -(1+a) g0 + c g3 ; g2 = a g0 - (1+c) g3
    e.g[0] = g0;
    e.g[3] = g3;
    e.g[1] = make_scalar3(-(1.0 + a) * g0.x + c * g3.x,
                          -(1.0 + a) * g0.y + c * g3.y,
                          -(1.0 + a) * g0.z + c * g3.z);
    e.g[2] = make_scalar3(a * g0.x - (1.0 + c) * g3.x,
                          a * g0.y - (1.0 + c) * g3.y,
                          a * g0.z - (1.0 + c) * g3.z);
    }

    } // anonymous namespace

// --------------------------------------------------------------------------- //
// ForceCompute                                                                //
// --------------------------------------------------------------------------- //
TISManyBodyForceCompute::TISManyBodyForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_energy(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TISManyBodyForceCompute" << endl;
    }

TISManyBodyForceCompute::~TISManyBodyForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying TISManyBodyForceCompute" << endl;
    }

void TISManyBodyForceCompute::setInteractions(pybind11::list interactions)
    {
    m_interactions.clear();
    m_interactions.reserve(pybind11::len(interactions));
    for (auto item : interactions)
        {
        pybind11::tuple t = item.cast<pybind11::tuple>();
        TISInteraction inter;
        inter.U0 = t[0].cast<double>();
        inter.has_gate = t[1].cast<bool>();
        inter.gate_i = t[2].cast<unsigned int>();
        inter.gate_j = t[3].cast<unsigned int>();
        double cutoff = t[4].cast<double>();
        inter.gate_cutoff2 = cutoff * cutoff;

        pybind11::list coords = t[5].cast<pybind11::list>();
        inter.coords.reserve(pybind11::len(coords));
        for (auto c : coords)
            {
            pybind11::tuple ct = c.cast<pybind11::tuple>();
            TISCoord coord;
            coord.kind = ct[0].cast<int>();
            pybind11::tuple idx = ct[1].cast<pybind11::tuple>();
            coord.n = (int)pybind11::len(idx);
            if (coord.n < 2 || coord.n > 4)
                throw runtime_error("TISManyBodyForceCompute: coord must have 2-4 indices");
            for (int a = 0; a < coord.n; a++)
                coord.tag[a] = idx[a].cast<unsigned int>();
            coord.k = ct[2].cast<double>();
            coord.q0 = ct[3].cast<double>();
            inter.coords.push_back(coord);
            }
        m_interactions.push_back(std::move(inter));
        }
    }

void TISManyBodyForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);

    // zero the force / per-particle energy array
    std::memset((void*)h_force.data, 0, sizeof(ForceReal4) * m_force.getNumElements());

    const BoxDim& box = m_pdata->getGlobalBox();
    const unsigned int N = m_pdata->getN();
    m_energy = 0.0;

    // scratch: local index of each particle in an interaction
    unsigned int loc[4 * 32];   // <= coords * 4; interactions here have <= 6 coords
    CoordEval evals[32];        // one per coordinate

    for (const TISInteraction& inter : m_interactions)
        {
        // distance gate: skip if the two gate particles are farther than cutoff
        if (inter.has_gate)
            {
            unsigned int gi = h_rtag.data[inter.gate_i];
            unsigned int gj = h_rtag.data[inter.gate_j];
            if (gi == NOT_LOCAL || gj == NOT_LOCAL)
                continue;
            Scalar3 pi = make_scalar3(h_pos.data[gi].x, h_pos.data[gi].y, h_pos.data[gi].z);
            Scalar3 pj = make_scalar3(h_pos.data[gj].x, h_pos.data[gj].y, h_pos.data[gj].z);
            Scalar3 d = box.minImage(v_sub(pi, pj));
            if (v_dot(d, d) > inter.gate_cutoff2)
                continue;
            }

        const size_t nc = inter.coords.size();

        // first pass: coordinate values + gradients -> denominator D
        double D = 1.0;
        for (size_t ci = 0; ci < nc; ci++)
            {
            const TISCoord& coord = inter.coords[ci];
            Scalar3 p[4];
            for (int a = 0; a < coord.n; a++)
                {
                unsigned int idx = h_rtag.data[coord.tag[a]];
                if (idx == NOT_LOCAL)
                    throw runtime_error("TISManyBodyForceCompute: interaction particle not local");
                p[a] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
                }
            CoordEval& e = evals[ci];
            if (coord.kind == TIS_DIST)
                eval_dist(p[0], p[1], box, e);
            else if (coord.kind == TIS_ANGLE)
                eval_angle(p[0], p[1], p[2], box, e);
            else
                eval_dihedral(p[0], p[1], p[2], p[3], box, e);

            double dq = (coord.kind == TIS_DIST) ? (e.q - coord.q0) : wrap(e.q - coord.q0);
            e.q = dq; // store the (wrapped) deviation for the second pass
            D += coord.k * dq * dq;
            }

        double U = inter.U0 / D;
        m_energy += U;
        double pref = U / D;

        // collect distinct local indices touched by this interaction (for the
        // per-particle energy share and the force scatter).
        int n_loc = 0;
        for (size_t ci = 0; ci < nc; ci++)
            {
            const TISCoord& coord = inter.coords[ci];
            for (int a = 0; a < coord.n; a++)
                {
                unsigned int idx = h_rtag.data[coord.tag[a]];
                bool seen = false;
                for (int q = 0; q < n_loc; q++)
                    if (loc[q] == idx)
                        {
                        seen = true;
                        break;
                        }
                if (!seen)
                    loc[n_loc++] = idx;
                }
            }
        double share = U / (double)n_loc;
        for (int q = 0; q < n_loc; q++)
            if (loc[q] < N)
                h_force.data[loc[q]].w += (ForceReal)share;

        // second pass: scatter forces.
        // dU/dq_a = -pref * 2 k_a dq_a ; force on x = -(dU/dq_a) * grad_x
        for (size_t ci = 0; ci < nc; ci++)
            {
            const TISCoord& coord = inter.coords[ci];
            const CoordEval& e = evals[ci];
            double dU_dq = -pref * 2.0 * coord.k * e.q; // e.q holds the deviation dq
            for (int a = 0; a < coord.n; a++)
                {
                unsigned int idx = h_rtag.data[coord.tag[a]];
                if (idx >= N)
                    continue;
                // f = -dU_dq * grad
                h_force.data[idx].x += (ForceReal)(-dU_dq * e.g[a].x);
                h_force.data[idx].y += (ForceReal)(-dU_dq * e.g[a].y);
                h_force.data[idx].z += (ForceReal)(-dU_dq * e.g[a].z);
                }
            }
        }
    }

namespace detail
    {
void export_TISManyBodyForceCompute(pybind11::module& m)
    {
    pybind11::class_<TISManyBodyForceCompute, ForceCompute, std::shared_ptr<TISManyBodyForceCompute>>(
        m,
        "TISManyBodyForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setInteractions", &TISManyBodyForceCompute::setInteractions)
        .def("getNumInteractions", &TISManyBodyForceCompute::getNumInteractions)
        .def("getEnergy", &TISManyBodyForceCompute::getEnergy);
    }
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
