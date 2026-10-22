/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <type_traits>

#if ALPAKA_OMP

namespace alpaka::onHost
{
    namespace internal
    {
        /** Number of OpenMP tasks used by exec::CpuOmpTasks.
         *
         * Read once from the environment variable ALPAKA_OMP_TASKS_NUM. If unset, invalid or zero the `num_tasks`
         * clause is omitted and the OpenMP runtime decides.
         */
        inline std::size_t getOmpTasksNumTasks()
        {
            static std::size_t const numTasks = []() -> std::size_t
            {
                char const* env = std::getenv("ALPAKA_OMP_TASKS_NUM");
                if(env == nullptr || *env == '\0')
                    return 0u;
                char* end = nullptr;
                auto const value = std::strtoull(env, &end, 10);
                return (end != env && *end == '\0') ? static_cast<std::size_t>(value) : 0u;
            }();
            return numTasks;
        }

        /** Enables debug output of exec::CpuOmpTasks if the environment variable ALPAKA_OMP_TASKS_DEBUG is set to a
         * value other than empty or "0".
         */
        inline bool isOmpTasksDebug()
        {
            static bool const enabled = []()
            {
                char const* env = std::getenv("ALPAKA_OMP_TASKS_DEBUG");
                return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
            }();
            return enabled;
        }
    } // namespace internal

    namespace cpu
    {
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        struct OmpTasks
        {
            constexpr OmpTasks(T_ThreadSpec threadSpec) : m_threadSpec{std::move(threadSpec)}
            {
                if(m_threadSpec.getNumThreads().product() != 1u)
                {
                    throw std::runtime_error("Thread block extent must be 1.");
                }
            }

            /** Execute the kernel bundle with OpenMP.
             *
             * @attention If this method is called from within an existing OpenMP parallel scope the function must be
             * called collectivly. All existing threads will execute the kernel bundle together and there will be no
             * thread synchronization after the function call.
             *
             * If the function is called from without beeing in a parallel OpenMP scope the blocks are distributed with an OpenMP taskloop.
             * The number of tasks can be set with the environment variable ALPAKA_OMP_TASKS_NUM.
             */
            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                using NumThreadsVecType = typename T_ThreadSpec::NumThreadsVecType;

                auto fn = [&kernelBundle, &dict, threadSpec = m_threadSpec]()
                {
                    auto const blockCount = threadSpec.getNumBlocks();
                    using ThreadIdxType = typename NumThreadsVecType::value_type;
                    ThreadIdxType const numBlocks = blockCount.product();

                    bool const debug = internal::isOmpTasksDebug();
                    /* Number of blocks executed per team thread, only used if debugging is enabled. The vector lives
                     * in the frame of the thread creating the tasks. */
                    std::vector<uint64_t> blocksPerThread;
                    if(debug)
                        blocksPerThread.assign(static_cast<std::size_t>(::omp_get_num_threads()), 0u);

                    /* Tasks can be executed by any thread of the team, therefore all block local state is created per
                     * block inside the task. */
                    auto runBlock = [&](ThreadIdxType i)
                    {
                        // copy from num blocks to derive correct index type
                        auto blockIdx = blockCount;
                        constexpr uint32_t simdWidth
                            = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});
                        auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared<simdWidth>{};

                        // dynamic shared mem
                        uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(threadSpec, kernelBundle);
                        auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                        auto const blockDynSharedMemBytesEntry
                            = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                        /* Only add dynamic shared memory objects if defined by the user, if not we will get a clean
                         * static assert if the kernel tries to access dynamic shared memory */
                        auto additionalDict = conditionalAppendDict<
                            trait::HasUserDefinedDynSharedMemBytes<T_ThreadSpec, ALPAKA_TYPEOF(kernelBundle)>::value>(
                            dict,
                            Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                        auto const blockLayerEntry = DictEntry{
                            layer::block,
                            onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}};
                        auto const threadLayerEntry
                            = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                        auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                        auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};
                        auto const warpSizeEntry = DictEntry{object::warpSize, std::integral_constant<uint32_t, 1u>{}};

                        auto acc = onAcc::Acc(joinDict(
                            Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry, warpSizeEntry},
                            additionalDict));

                        blockIdx = mapToND(blockCount, i);
                        if(debug)
                        {
                            auto const tid = static_cast<std::size_t>(::omp_get_thread_num());
                            uint64_t* counter = &blocksPerThread[tid];
#    pragma omp atomic
                            *counter += 1u;
                        }
                        kernelBundle(acc);
                    };

                    auto const numTasks = static_cast<ThreadIdxType>(internal::getOmpTasksNumTasks());

                    /* One thread creates the tasks, all other threads of the team execute them. The taskloop has an
                     * implicit taskgroup, the creating thread therefore returns after all blocks are finished. */
#    pragma omp single nowait
                    {
                        if(debug)
                            std::fprintf(
                                stderr,
                                "[alpaka OmpTasks] numBlocks=%llu ALPAKA_OMP_TASKS_NUM=%llu%s team_threads=%d "
                                "max_threads=%d creator_thread=%d in_final=%d\n",
                                static_cast<unsigned long long>(numBlocks),
                                static_cast<unsigned long long>(numTasks),
                                numTasks == 0 ? " (unset, runtime decides)" : "",
                                ::omp_get_num_threads(),
                                ::omp_get_max_threads(),
                                ::omp_get_thread_num(),
                                ::omp_in_final());
                        if(numTasks != 0)
                        {
#    pragma omp taskloop default(shared) num_tasks(numTasks)
                            for(ThreadIdxType i = 0; i < numBlocks; ++i)
                                runBlock(i);
                        }
                        else
                        {
#    pragma omp taskloop default(shared)
                            for(ThreadIdxType i = 0; i < numBlocks; ++i)
                                runBlock(i);
                        }
                        if(debug)
                        {
                            std::size_t activeThreads = 0u;
                            for(auto const c : blocksPerThread)
                                activeThreads += c != 0u ? 1u : 0u;
                            std::fprintf(
                                stderr,
                                "[alpaka OmpTasks] finished: %zu of %zu threads executed blocks. blocks per thread:",
                                activeThreads,
                                blocksPerThread.size());
                            for(auto const c : blocksPerThread)
                                std::fprintf(stderr, " %llu", static_cast<unsigned long long>(c));
                            std::fprintf(stderr, "\n");
                        }
                    }
                };

                if(internal::isOmpTasksDebug())
                    std::fprintf(
                        stderr,
                        "[alpaka OmpTasks] launch: in_parallel=%d (%s)\n",
                        ::omp_in_parallel(),
                        ::omp_in_parallel() != 0 ? "collective call, only one thread creates the tasks"
                                                 : "opening own parallel region");
                if(::omp_in_parallel() != 0)
                {
                    /* We are already in a OpenMP parllel section, do not start a new section to avoid nested
                     * parallelism which is typical slow.
                     * There is no synchronization after the function execution happen, the caller is responsible for
                     * it.
                     */
                    fn();
                }
                else
                {
#    pragma omp parallel
                    fn();
                }
            }

        private:
            T_ThreadSpec m_threadSpec;
        };
    } // namespace cpu

    inline auto makeAcc(
        alpaka::onHost::concepts::ThreadSpec auto const& threadSpec,
        uint32_t /* numaIdx */,
        bool /* setThreadAffinity */) requires std::same_as<ALPAKA_TYPEOF(threadSpec.getExecutor()), exec::CpuOmpTasks>
    {
        // thread affinity is controlled by the OpenMP runtime (OMP_PLACES, OMP_PROC_BIND)
        return cpu::OmpTasks(threadSpec);
    }
} // namespace alpaka::onHost

#endif
