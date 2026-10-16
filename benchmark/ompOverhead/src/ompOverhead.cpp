/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** Compares the launch/scheduling overhead of exec::cpuOmpBlocks (omp for) and exec::cpuOmpTasks (omp taskloop).
 *
 * Output is CSV on stdout (one line per executor/kernel/block count), diagnostics go to stderr.
 * The number of tasks of cpuOmpTasks is read from ALPAKA_OMP_TASKS_NUM once per process, use run.sh to sweep it.
 *
 * Kernels (one thread per block):
 *  - empty:      does nothing, measures pure launch + block scheduling overhead
 *  - balanced:   every block writes its index, tiny uniform work
 *  - imbalanced: work of block b is proportional to (b % 64), static scheduling suffers, tasks should help
 */

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace alpaka;

struct EmptyKernel
{
    ALPAKA_FN_ACC auto operator()(auto const&) const -> void
    {
    }
};

struct BalancedKernel
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, alpaka::concepts::IMdSpan auto out) const -> void
    {
        auto const b = acc[alpaka::layer::block].idx()[0];
        out[b] = static_cast<float>(b);
    }
};

struct ImbalancedKernel
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, alpaka::concepts::IMdSpan auto out, uint32_t workUnit) const -> void
    {
        auto const b = acc[alpaka::layer::block].idx()[0];
        uint32_t const iters = static_cast<uint32_t>(b % 64u) * workUnit;
        float x = static_cast<float>(b);
        for(uint32_t i = 0; i < iters; ++i)
            x = x * 0.999999f + 1.0f;
        out[b] = x;
    }
};

template<typename T_Fn>
void measure(
    std::string const& execName,
    std::string const& kernelName,
    size_t numBlocks,
    int reps,
    int warmup,
    double budgetSec,
    T_Fn&& launch)
{
    std::cerr << "  " << execName << " " << kernelName << " blocks=" << numBlocks << std::endl;
    /* Stop early once the time budget is used up (at least one measured repetition), so heavy configurations
     * (many blocks, few threads) do not take forever. Warmup is cut to one run if a single launch is slow. */
    std::vector<double> times;
    double spent = 0.0;
    for(int i = 0; i < warmup + reps; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        launch();
        auto end = std::chrono::steady_clock::now();
        double const us = std::chrono::duration<double, std::micro>(end - start).count();
        spent += us * 1e-6;
        if(i >= warmup)
            times.push_back(us);
        else if(us * 1e-6 > budgetSec / 10.0)
            warmup = i + 1; // slow launch: one warmup is enough
        if(i >= warmup && spent > budgetSec)
            break;
    }
    std::sort(times.begin(), times.end());
    double const minT = times.front();
    double const medT = times[times.size() / 2];
    char const* tasksEnv = std::getenv("ALPAKA_OMP_TASKS_NUM");
    char const* threadsEnv = std::getenv("OMP_NUM_THREADS");
    std::cout << execName << "," << (tasksEnv ? tasksEnv : "unset") << "," << (threadsEnv ? threadsEnv : "unset")
              << "," << kernelName << "," << numBlocks << "," << minT << "," << medT << ","
              << (medT * 1000.0 / static_cast<double>(numBlocks)) << std::endl;
}

void runExecutor(auto const exec, int reps, int warmup, double budgetSec, size_t maxBlocks, uint32_t workUnit)
{
    auto devSelector = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu});
    onHost::Device dev = devSelector.makeDevice(0);
    // blocking queue: no callback thread, the launch cost is not hidden
    onHost::Queue queue = dev.makeQueue(queueKind::blocking);
    std::string const execName = onHost::demangledName(exec);

    auto out = onHost::alloc<float>(dev, maxBlocks);

    for(size_t numBlocks = 1; numBlocks <= maxBlocks; numBlocks *= 10)
    {
        auto frameSpec = onHost::FrameSpec{numBlocks, size_t{1}, exec};

        auto const empty = KernelBundle{EmptyKernel{}};
        measure(execName, "empty", numBlocks, reps, warmup, budgetSec, [&] { queue.enqueue(frameSpec, empty); });

        auto const balanced = KernelBundle{BalancedKernel{}, out};
        measure(execName, "balanced", numBlocks, reps, warmup, budgetSec, [&] { queue.enqueue(frameSpec, balanced); });

        auto const imbalanced = KernelBundle{ImbalancedKernel{}, out, workUnit};
        measure(execName, "imbalanced", numBlocks, reps, warmup, budgetSec, [&] { queue.enqueue(frameSpec, imbalanced); });
    }
}

auto main(int argc, char* argv[]) -> int
{
    int reps = 50;
    int warmup = 5;
    size_t maxBlocks = 1000000;
    uint32_t workUnit = 20;
    double budgetSec = 1.0;
    int opt;
    while((opt = getopt(argc, argv, "hr:w:m:u:t:")) != -1)
    {
        switch(opt)
        {
        case 'r':
            reps = std::stoi(optarg);
            break;
        case 'w':
            warmup = std::stoi(optarg);
            break;
        case 'm':
            maxBlocks = std::stoul(optarg);
            break;
        case 't':
            budgetSec = std::stod(optarg);
            break;
        case 'u':
            workUnit = static_cast<uint32_t>(std::stoul(optarg));
            break;
        default:
            std::cerr << "usage: " << argv[0]
                      << " [-r repetitions] [-w warmup] [-m max blocks (powers of 10)] [-u work units per imbalance step] [-t time budget per measurement in s]\n"
                      << "env: ALPAKA_OMP_TASKS_NUM (cpuOmpTasks number of tasks), OMP_NUM_THREADS\n";
            return opt == 'h' ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    std::cout << "executor,tasks_num,omp_threads,kernel,blocks,min_us,median_us,median_ns_per_block" << std::endl;
    runExecutor(exec::cpuOmpBlocks, reps, warmup, budgetSec, maxBlocks, workUnit);
    runExecutor(exec::cpuOmpTasks, reps, warmup, budgetSec, maxBlocks, workUnit);
    return EXIT_SUCCESS;
}
