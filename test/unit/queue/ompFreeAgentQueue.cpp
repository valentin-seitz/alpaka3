/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

/** This test is testing the experimental queue kind ompFreeAgent which executes all operations as OpenMP tasks.
 *
 * The queue is **NOT** part of official alpaka, therefore it must be included separately.
 * The test is skipped if alpaka tests are build without OpenMP support.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/api/host/OmpFreeAgentQueue.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mutex>
#include <vector>

using namespace alpaka;

using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, std::make_tuple(exec::cpuOmpBlockTasks)))>;

#if ALPAKA_OMP

//! out[i] = out[i] * factor + summand
struct AxpbKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t factor, uint32_t summand) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
            out[i] = out[i] * factor + summand;
    }
};

//! out[i] = in[i] + summand
struct AddKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto in, uint32_t summand) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
            out[i] = in[i] + summand;
    }
};

bool allEqual(auto queue, auto& dBuffer, uint32_t expected)
{
    auto hBuffer = onHost::allocHostLike(dBuffer);
    onHost::memcpy(queue, hBuffer, dBuffer);
    onHost::wait(queue);

    bool result = true;
    meta::ndLoopIncIdx(hBuffer.getExtents(), [&](auto const& idx) { result = result && hBuffer[idx] == expected; });
    return result;
}

TEMPLATE_LIST_TEST_CASE("ompFreeAgent in-order kernels", "[queue][ompFreeAgent]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue(queueKind::ompFreeAgent);

    auto extents = Vec{1031u};
    auto frameExtent = Vec{16u};
    auto frameSpec = onHost::FrameSpec{divCeil(extents, frameExtent), frameExtent, exec};
    auto dBuffer = onHost::alloc<uint32_t>(device, extents);

    // the result is only correct if all kernels are executed in order
    onHost::memset(queue, dBuffer, 0x00);
    uint32_t expected = 0u;
    for(uint32_t k = 0u; k < 20u; ++k)
    {
        queue.enqueue(frameSpec, AxpbKernel{}, dBuffer, 2u, k);
        expected = expected * 2u + k;
    }
    CHECK(allEqual(queue, dBuffer, expected));
    CHECK(queue.isEmpty());
}

TEMPLATE_LIST_TEST_CASE("ompFreeAgent host functions", "[queue][ompFreeAgent]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    onHost::Queue queue = device.makeQueue(queueKind::ompFreeAgent);

    std::vector<uint32_t> order;
    for(uint32_t i = 0u; i < 100u; ++i)
        queue.enqueueHostFn([&order, i] { order.push_back(i); });
    onHost::wait(queue);

    REQUIRE(order.size() == 100u);
    bool inOrder = true;
    for(uint32_t i = 0u; i < 100u; ++i)
        inOrder = inOrder && order[i] == i;
    CHECK(inOrder);
    CHECK(queue.isEmpty());
}

TEMPLATE_LIST_TEST_CASE("ompFreeAgent events between queues", "[queue][ompFreeAgent]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue producer = device.makeQueue(queueKind::ompFreeAgent);
    onHost::Queue consumer = device.makeQueue(queueKind::ompFreeAgent);
    onHost::Queue normalQueue = device.makeQueue(queueKind::nonBlocking);

    auto extents = Vec{64u, 33u};
    auto frameExtent = Vec{4u, 8u};
    auto frameSpec = onHost::FrameSpec{divCeil(extents, frameExtent), frameExtent, exec};
    auto dIn = onHost::alloc<uint32_t>(device, extents);
    auto dOut = onHost::alloc<uint32_t>(device, extents);
    auto dOut2 = onHost::alloc<uint32_t>(device, extents);

    auto event = device.makeEvent();

    for(uint32_t round = 0u; round < 10u; ++round)
    {
        // producer: dIn = round + 1, slowed down by many kernel launches
        onHost::memset(producer, dIn, 0x00);
        for(uint32_t k = 0u; k <= round; ++k)
            producer.enqueue(frameSpec, AxpbKernel{}, dIn, 1u, 1u);
        producer.enqueue(event);

        // consumer queue of the same scheduler waits via a task dependency
        consumer.waitFor(event);
        consumer.enqueue(frameSpec, AddKernel{}, dOut, dIn, 100u);

        // a normal queue waits by blocking on the event
        normalQueue.waitFor(event);
        normalQueue.enqueue(frameSpec, AddKernel{}, dOut2, dIn, 200u);

        CHECK(allEqual(consumer, dOut, round + 1u + 100u));
        CHECK(allEqual(normalQueue, dOut2, round + 1u + 200u));
        onHost::wait(producer);
    }

    // event enqueued in a normal queue, waited by a task queue
    onHost::memset(normalQueue, dIn, 0x00);
    normalQueue.enqueueHostFn([] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); });
    normalQueue.enqueue(frameSpec, AxpbKernel{}, dIn, 1u, 7u);
    normalQueue.enqueue(event);
    consumer.waitFor(event);
    consumer.enqueue(frameSpec, AddKernel{}, dOut, dIn, 1u);
    CHECK(allEqual(consumer, dOut, 8u));

    onHost::wait(event);
    CHECK(event.isComplete());
}

TEMPLATE_LIST_TEST_CASE("ompFreeAgent many queues", "[queue][ompFreeAgent]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    constexpr uint32_t numQueues = 8u;
    auto extents = Vec{257u};
    auto frameExtent = Vec{8u};
    auto frameSpec = onHost::FrameSpec{divCeil(extents, frameExtent), frameExtent, exec};

    using QueueType = ALPAKA_TYPEOF(device.makeQueue(queueKind::ompFreeAgent));
    using BufferType = ALPAKA_TYPEOF(onHost::alloc<uint32_t>(device, extents));
    std::vector<QueueType> queues;
    std::vector<BufferType> buffers;
    for(uint32_t q = 0u; q < numQueues; ++q)
    {
        queues.push_back(device.makeQueue(queueKind::ompFreeAgent));
        buffers.push_back(onHost::alloc<uint32_t>(device, extents));
        onHost::memset(queues[q], buffers[q], 0x00);
    }
    // interleave the launches of all queues
    for(uint32_t k = 0u; k < 50u; ++k)
        for(uint32_t q = 0u; q < numQueues; ++q)
            queues[q].enqueue(frameSpec, AxpbKernel{}, buffers[q], 1u, q + 1u);

    // deferred allocation released while operations are pending
    {
        auto tmp = onHost::allocDeferred<uint32_t>(queues[0], extents);
        onHost::memset(queues[0], tmp, 0x00);
    }

    device.wait();
    for(uint32_t q = 0u; q < numQueues; ++q)
    {
        CHECK(queues[q].isEmpty());
        CHECK(allEqual(queues[q], buffers[q], 50u * (q + 1u)));
    }
}

TEMPLATE_LIST_TEST_CASE("ompFreeAgent queue destruction waits", "[queue][ompFreeAgent]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);

    std::atomic<uint32_t> counter{0u};
    {
        onHost::Queue queue = device.makeQueue(queueKind::ompFreeAgent);
        for(uint32_t i = 0u; i < 10u; ++i)
            queue.enqueueHostFn(
                [&counter]
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    ++counter;
                });
    }
    CHECK(counter.load() == 10u);
}
#else
TEST_CASE("ompFreeAgent", "[queue][ompFreeAgent]")
{
    WARN("no OpenMP available: Disable ompFreeAgent test");
    SUCCEED("SKIP ompFreeAgent test");
}
#endif
