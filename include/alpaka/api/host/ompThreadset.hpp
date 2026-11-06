/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file Helper for the OpenMP 6.0 `threadset` clause.
 *
 * ALPAKA_OMP_THREADSET_POOL expands to `threadset(omp_pool)` if the compiler supports OpenMP 6.0, else it is empty.
 * Tasks created with `threadset(omp_pool)` can be executed by free-agent threads, i.e. threads which are not part of
 * a team. The macro can be predefined to override the detection.
 */

#ifndef ALPAKA_OMP_THREADSET_POOL
#    if defined(_OPENMP) && _OPENMP >= 202411
#        define ALPAKA_OMP_THREADSET_POOL threadset(omp_pool)
#    else
#        define ALPAKA_OMP_THREADSET_POOL
#    endif
#endif
