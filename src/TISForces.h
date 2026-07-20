// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file TISForces.h
    \brief Generic many-body ForceCompute for the two TIS custom terms.

    Both the single-stranded stacking and the hydrogen-bonding term share the
    rational "many-body" form (MODEL.md Eq. 9 / 13):

        U = U0 / ( 1 + Sum_a k_a (q_a - q_a0)^2 )

    where each geometric coordinate q_a is a distance, bond angle, or dihedral
    among a handful of point particles. This one ForceCompute holds a list of
    such interactions (built on the Python side, tag-indexed) and evaluates the
    energy and the exact Cartesian gradient of each. An interaction may carry an
    optional distance GATE (two tags + cutoff): if |B_i - B_j| > cutoff the
    interaction is skipped this step (this is how hydrogen bonding switches on
    and off).

    This is the compiled replacement for the pure-Python
    tis.custom_forces._ManyBodyCustomForce evaluator; the physics is ported
    verbatim (see tis/custom_forces.py, the validated reference).
*/

#pragma once

#include "hoomd/ForceCompute.h"

#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Coordinate kinds.
enum TISCoordKind
    {
    TIS_DIST = 0,
    TIS_ANGLE = 1,
    TIS_DIHEDRAL = 2
    };

//! A single geometric coordinate (distance, angle, or dihedral).
struct TISCoord
    {
    int kind;             //!< TISCoordKind
    int n;                //!< number of participating particles (2, 3, or 4)
    unsigned int tag[4];  //!< global TAGS of the participating particles
    double k;             //!< force constant
    double q0;            //!< reference value
    };

//! One rational many-body interaction (one well depth + its coordinates).
struct TISInteraction
    {
    double U0;                     //!< well depth
    bool has_gate;                 //!< whether a distance gate is applied
    unsigned int gate_i, gate_j;   //!< TAGS of the two gate particles
    double gate_cutoff2;           //!< squared distance cutoff for the gate
    std::vector<TISCoord> coords;  //!< the interaction's coordinates
    };

//! Generic TIS many-body force (stacking + hydrogen bonding).
class PYBIND11_EXPORT TISManyBodyForceCompute : public ForceCompute
    {
    public:
    TISManyBodyForceCompute(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~TISManyBodyForceCompute();

    //! Replace the interaction list from Python (nested list, see cpp_forces.py).
    void setInteractions(pybind11::list interactions);

    //! Number of interactions currently held.
    size_t getNumInteractions() const { return m_interactions.size(); }

    //! Total potential energy (double accumulator) from the last computeForces().
    double getEnergy() { return m_energy; }

#ifdef ENABLE_MPI
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    std::vector<TISInteraction> m_interactions; //!< interaction list (tag-indexed)
    double m_energy;                            //!< accumulated energy (double)

    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_TISManyBodyForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
