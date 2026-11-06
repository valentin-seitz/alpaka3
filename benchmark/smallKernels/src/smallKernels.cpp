/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

/** Many short kernels on several independent queues.
 *
 * Compares non-blocking queues, where each kernel opens its own OpenMP parallel region, with the experimental queue
 * kind ompFreeAgent, where each kernel is a task and the blocks are tasks executed by free-agent threads.
 *
 * usage: smallKernels [numQueues] [kernelsPerQueue] [numElements] [numFramesPerKernel] [repetitions]
 * output: CSV line per queue kind
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/api/host/OmpFreeAgentQueue.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace alpaka;
using Vec1D = Vec<uint32_t, 1u>;

//! out[i] = out[i] * 0.5 + 1, a few iterations per element to have some work
struct SmallKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto data, uint32_t iterations) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{data.getExtents()}))
        {
            float v = data[i];
            for(uint32_t k = 0u; k < iterations; ++k)
                v = v * 0.5f + 1.0f;
            data[i] = v;
        }
    }
};

double run(
    auto& device,
    auto queueKind,
    uint32_t numQueues,
    uint32_t kernelsPerQueue,
    uint32_t numElements,
    uint32_t numFrames,
    uint32_t repetitions)
{
    using QueueType = ALPAKA_TYPEOF(device.makeQueue(queueKind));
    using BufferType = ALPAKA_TYPEOF(onHost::alloc<float>(device, Vec1D{numElements}));
    std::vector<QueueType> queues;
    std::vector<BufferType> buffers;
    for(uint32_t q = 0u; q < numQueues; ++q)
    {
        queues.push_back(device.makeQueue(queueKind));
        buffers.push_back(onHost::alloc<float>(device, Vec1D{numElements}));
        onHost::memset(queues[q], buffers[q], 0x00);
    }

    auto frameExtent = divCeil(Vec1D{numElements}, Vec1D{numFrames});
    auto frameSpec = onHost::FrameSpec{Vec1D{numFrames}, frameExtent, exec::cpuOmpBlockTasks};

    std::vector<double> times;
    // first repetition is a warm-up
    for(uint32_t r = 0u; r < repetitions + 1u; ++r)
    {
        device.wait();
        auto const begin = std::chrono::steady_clock::now();
        for(uint32_t k = 0u; k < kernelsPerQueue; ++k)
            for(uint32_t q = 0u; q < numQueues; ++q)
                queues[q].enqueue(frameSpec, SmallKernel{}, buffers[q], 16u);
        device.wait();
        auto const end = std::chrono::steady_clock::now();
        if(r != 0u)
            times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2u];
}

int main(int argc, char** argv)
{
    auto arg = [&](int i, uint32_t def) { return argc > i ? static_cast<uint32_t>(std::atoi(argv[i])) : def; };
    uint32_t const numQueues = arg(1, 4u);
    uint32_t const kernelsPerQueue = arg(2, 200u);
    uint32_t const numElements = arg(3, 1u << 14u);
    uint32_t const numFrames = arg(4, 8u);
    uint32_t const repetitions = arg(5, 5u);

    auto selector = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu});
    onHost::Device device = selector.makeDevice(0);

    std::cout << "queueKind,numQueues,kernelsPerQueue,numElements,numFrames,median_ms,us_per_kernel" << std::endl;
    auto report = [&](std::string const& name, double ms)
    {
        std::cout << name << "," << numQueues << "," << kernelsPerQueue << "," << numElements << "," << numFrames
                  << "," << ms << "," << ms * 1000.0 / (numQueues * kernelsPerQueue) << std::endl;
    };
    report(
        "nonBlocking",
        run(device, queueKind::nonBlocking, numQueues, kernelsPerQueue, numElements, numFrames, repetitions));
    report(
        "ompFreeAgent",
        run(device, queueKind::ompFreeAgent, numQueues, kernelsPerQueue, numElements, numFrames, repetitions));
    return EXIT_SUCCESS;
}
