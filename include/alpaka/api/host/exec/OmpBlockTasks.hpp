/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if ALPAKA_OMP

namespace alpaka::onHost
{
    namespace cpu
    {
        /** Executes each thread block as an OpenMP task.
         *
         * The number of tasks is equal to the number of thread blocks, the task granularity is therefore controlled
         * by the frame extent (or the work a kernel does per block). If the frame extent is a compile time vector the
         * granularity is fixed at compile time, otherwise it can be chosen at runtime.
         *
         * The executor never changes the thread affinity, the placement of threads is left to the OpenMP runtime,
         * e.g. controlled by OMP_PROC_BIND and OMP_PLACES.
         */
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        struct OmpBlockTasks
        {
            using NumThreadsVecType = typename T_ThreadSpec::NumThreadsVecType;

            constexpr OmpBlockTasks(T_ThreadSpec threadSpec, uint32_t numaIdx, bool setThreadAffinity)
                : m_threadSpec{std::move(threadSpec)}
            {
                // thread affinity is handled by the OpenMP runtime
                alpaka::unused(numaIdx, setThreadAffinity);
                if(m_threadSpec.getNumThreads().product() != 1u)
                {
                    throw std::runtime_error("Thread block extent must be 1.");
                }
            }

            /** Execute the kernel bundle with one OpenMP task per thread block.
             *
             * If the method is called outside of an OpenMP parallel region, a new parallel region is opened and one
             * thread creates the tasks.
             * If the method is called from within an OpenMP parallel region (e.g. from within a task or a single
             * construct), the calling thread creates the tasks and the tasks are executed by the threads of the
             * enclosing team. The call is not collective, each calling thread executes the full kernel.
             *
             * The method returns after all tasks of the kernel are finished.
             */
            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                auto const blockCount = m_threadSpec.getNumBlocks();

                constexpr uint32_t simdWidth
                    = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});

                auto executeBlock = [&](auto const linearBlockIdx)
                {
                    auto const blockIdx = mapToND(blockCount, linearBlockIdx);

                    auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared<simdWidth>{};

                    auto const blockLayerEntry
                        = DictEntry{layer::block, onAcc::cpu::GenericLayer{std::cref(blockIdx), blockCount}};
                    auto const threadLayerEntry = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                    auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                    auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                    // dynamic shared mem
                    uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(m_threadSpec, kernelBundle);
                    auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                    auto const blockDynSharedMemBytesEntry
                        = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                    /* Only add dynamic shared memory objects if defined by the user, if not we will get a clean static
                     * assert if the kernel tries to access dynamic shared memory */
                    auto additionalDict = conditionalAppendDict<
                        trait::HasUserDefinedDynSharedMemBytes<T_ThreadSpec, ALPAKA_TYPEOF(kernelBundle)>::value>(
                        dict,
                        Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                    auto const warpSizeEntry = DictEntry{object::warpSize, std::integral_constant<uint32_t, 1u>{}};

                    auto acc = onAcc::Acc(joinDict(
                        Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry, warpSizeEntry},
                        additionalDict));

                    kernelBundle(acc);
                };

                auto spawnTasks = [&]()
                {
                    using ThreadIdxType = typename NumThreadsVecType::value_type;
                    ThreadIdxType const linearNumBlocks = blockCount.product();

                    /* Explicit tasks are used instead of `omp taskloop grainsize(1)`: for a large number of blocks
                     * libgomp handles the taskloop as one undeferred batch and allocates the state of all tasks on
                     * the stack of the encountering thread, which overflows the stack. Explicit tasks are throttled by
                     * the runtime one by one. The taskgroup guarantees that all tasks (including nested tasks created
                     * by the kernel) are finished.
                     */
#    pragma omp taskgroup
                    {
                        for(ThreadIdxType i = 0; i < linearNumBlocks; ++i)
                        {
#    pragma omp task default(shared) firstprivate(i) threadset(omp_pool)
                            executeBlock(i);
                        }
                    }
                };

                if(::omp_in_parallel() != 0)
                    spawnTasks();
                else
                {
#    pragma omp parallel
#    pragma omp single
                    spawnTasks();
                }
            }

        private:
            T_ThreadSpec m_threadSpec;
        };
    } // namespace cpu

    inline auto makeAcc(
        alpaka::onHost::concepts::ThreadSpec auto const& threadSpec,
        uint32_t numaIdx,
        bool setThreadAffinity) requires std::same_as<ALPAKA_TYPEOF(threadSpec.getExecutor()), exec::CpuOmpBlockTasks>
    {
        return cpu::OmpBlockTasks(threadSpec, numaIdx, setThreadAffinity);
    }
} // namespace alpaka::onHost

#endif
