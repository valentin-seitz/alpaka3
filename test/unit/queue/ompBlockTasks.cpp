/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

/** This test is testing the executor cpuOmpBlockTasks which executes each thread block as an OpenMP task.
 *
 * The kernel is enqueued once outside of an OpenMP parallel region and once from within a single construct of a
 * parallel region, where the tasks are executed by the enclosing team.
 * The test is skipped if alpaka tests are build without OpenMP support.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, std::make_tuple(exec::cpuOmpBlockTasks)))>;

#if ALPAKA_OMP
/** The number of threads used for the OpenMP parallel section */
constexpr uint32_t numOmpThreads = 3;

struct CountKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto frameExtent, auto counter) const
    {
        auto const extents = counter.getExtents();
        constexpr uint32_t dim = ALPAKA_TYPEOF(extents)::dim();
        for(auto frameOffset : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{Vec<uint32_t, dim>::fill(0u), extents, frameExtent}))
        {
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent}))
            {
                auto const idx = frameOffset + i;
                bool inside = true;
                for(uint32_t d = 0u; d < dim; ++d)
                    inside = inside && idx[d] < extents[d];
                if(inside)
                    onAcc::atomicAdd(acc, &counter[idx], 1u);
            }
        }
    }
};

void checkCounter(auto queue, auto& dCounter, concepts::Vector auto extents)
{
    auto hCounter = onHost::allocHostLike(dCounter);
    onHost::memcpy(queue, hCounter, dCounter);
    onHost::wait(queue);

    bool allOne = true;
    meta::ndLoopIncIdx(extents, [&](auto const& idx) { allOne = allOne && hCounter[idx] == 1u; });
    CHECK(allOne);
}

void runTest(auto queue, auto exec, concepts::Vector auto extents, concepts::Vector auto frameExtent)
{
    auto numFrames = divCeil(extents, frameExtent);
    auto frameSpec = onHost::FrameSpec{numFrames, frameExtent, exec};
    auto dCounter = onHost::alloc<uint32_t>(queue.getDevice(), extents);

    // outside of an OpenMP parallel region
    onHost::memset(queue, dCounter, 0x00);
    queue.enqueue(frameSpec, CountKernel{}, frameExtent, dCounter);
    checkCounter(queue, dCounter, extents);

    // the tasks are created by one thread and executed by the enclosing team
    onHost::memset(queue, dCounter, 0x00);
    onHost::wait(queue);
#    pragma omp parallel num_threads(numOmpThreads)
#    pragma omp single
    queue.enqueue(frameSpec, CountKernel{}, frameExtent, dCounter);
    checkCounter(queue, dCounter, extents);
}

TEMPLATE_LIST_TEST_CASE("OmpBlockTasks", "[queue][OmpBlockTasks]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    // runtime frame extents
    runTest(queue, exec, Vec{997u}, Vec{10u});
    runTest(queue, exec, Vec{101u, 103u}, Vec{7u, 5u});
    runTest(queue, exec, Vec{43u, 47u, 53u}, Vec{4u, 4u, 4u});
    // compile time frame extents
    runTest(queue, exec, Vec{997u}, CVec<uint32_t, 32u>{});
    runTest(queue, exec, Vec{101u, 103u}, CVec<uint32_t, 8u, 8u>{});
}
#else
TEST_CASE("OmpBlockTasks", "[queue][OmpBlockTasks]")
{
    WARN("no OpenMP available: Disable OmpBlockTasks test");
    SUCCEED("SKIP OmpBlockTasks test");
}
#endif
