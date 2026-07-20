// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file MixedPrecisionCompat.h
    \brief Compatibility shim for building against both upstream HOOMD-blue
           and the mixed-precision (sloptimize) fork.

    When HOOMD_HAS_FORCEREAL is defined (sloptimize fork), ForceReal and related
    types/functions are already available from HOOMDMath.h and BoxDim.h.

    When building against upstream HOOMD-blue (which lacks ForceReal), this header
    provides aliases so that ForceReal == Scalar and all helper functions
    forward to their Scalar equivalents.
*/

#pragma once

#include "hoomd/HOOMDMath.h"

#ifndef HOOMD_HAS_FORCEREAL

namespace hoomd
    {
// Upstream HOOMD: ForceReal types don't exist — alias to Scalar
typedef Scalar ForceReal;
typedef Scalar2 ForceReal2;
typedef Scalar3 ForceReal3;
typedef Scalar4 ForceReal4;
    } // end namespace hoomd

// Alias helper functions as preprocessor macros (avoids HOSTDEVICE issues)
#define make_forcereal3 hoomd::make_scalar3
#define make_forcereal4 hoomd::make_scalar4

#endif // HOOMD_HAS_FORCEREAL
