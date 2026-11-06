/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This contains an experimental host queue executing all operations as OpenMP tasks.
 *
 * @attention This file is **NOT** included with `#include <alpaka/alpaka.hpp>`
 */

#pragma once

#include "alpaka/api/host/Device.hpp"
#include "alpaka/api/host/Queue.hpp"
#include "alpaka/api/host/ompThreadset.hpp"
#include "alpaka/core/TaskQueue.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/unused.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace alpaka
{
    namespace queueKind
    {
        /** Experimental non-blocking queue executing all operations as OpenMP tasks.
         *
         * All queues of this kind created for the same device share one scheduler. The scheduler owns one thread
         * which creates an OpenMP task with `depend` clauses for each operation:
         *   - operations of one queue are executed in order (the queue is a dependency token)
         *   - operations of different queues can overlap, there is no parallel region and no barrier per kernel
         *   - waiting for an event which was enqueued in a queue of this kind on the same device is a task
         *     dependency, no thread is blocked
         *
         * The tasks are created with `threadset(omp_pool)` if the compiler supports OpenMP 6.0, they are executed
         * by free-agent threads. The number of free-agent threads is controlled by the OpenMP runtime, e.g. via
         * OMP_THREADS_RESERVE. If the runtime has no free-agent threads all tasks are executed by the scheduler
         * thread.
         *
         * Kernels should use the executor exec::cpuOmpBlockTasks, each thread block is executed as a task.
         *
         * Limitations of the prototype:
         *   - exceptions thrown by an operation are forwarded to the future of the operation only
         *   - thread affinity is not set by alpaka
         *   - waiting for a queue or event from within an operation can deadlock if the runtime has no free-agent
         *     threads
         *
         * The queue can only be used if the dependency OpenMP is available, e.g. by setting the CMake option
         * ALPAKA_DEP_OMP=ON or using the required compiler flags to activate OpenMP.
         */
        struct OmpFreeAgent : category::QueueKind
        {
            static constexpr auto isBlocking()
            {
                return false;
            }

            static std::string getName()
            {
                return "OmpFreeAgent";
            }
        };

        /** @copydoc OmpFreeAgent */
        constexpr auto ompFreeAgent = OmpFreeAgent{};
    } // namespace queueKind
} // namespace alpaka

#if ALPAKA_OMP

namespace alpaka::onHost::cpu
{
    /** Creates OpenMP tasks for operations of task queues.
     *
     * One thread (the generator) creates all tasks. This is required because `depend` clauses only order sibling
     * tasks, i.e. tasks created by the same task region. The generator thread is not part of a parallel region, the
     * tasks are executed by free-agent threads.
     */
    class OmpTaskScheduler
    {
    public:
        //! State of one queue, the address of token is the dependency token of the queue.
        struct QueueState
        {
            char token = 0;
            //! number of operations submitted and not finished
            std::atomic<uint32_t> numPending{0u};
        };

        struct Operation
        {
            std::unique_ptr<core::TaskQueue::Task> task;
            std::promise<void> promise;
            std::shared_ptr<QueueState> queue;
            void const* waitToken = nullptr;
            void* signalToken = nullptr;
        };

        OmpTaskScheduler() : m_state(std::make_shared<State>())
        {
            m_thread = std::thread([state = m_state]() { generatorLoop(*state); });
        }

        ~OmpTaskScheduler()
        {
            {
                std::lock_guard<std::mutex> lk(m_state->mutex);
                m_state->stop = true;
            }
            m_state->cond.notify_one();
            if(std::this_thread::get_id() == m_thread.get_id())
            {
                // The last reference was released by an operation executed by the generator thread.
                m_thread.detach();
            }
            else
                m_thread.join();
        }

        OmpTaskScheduler(OmpTaskScheduler const&) = delete;
        OmpTaskScheduler& operator=(OmpTaskScheduler const&) = delete;

        /** Get the scheduler for a device.
         *
         * The scheduler is created on first use and destroyed after the last queue using it is destroyed.
         */
        static std::shared_ptr<OmpTaskScheduler> get(void const* deviceId)
        {
            static std::mutex registryMutex;
            static std::map<void const*, std::weak_ptr<OmpTaskScheduler>> registry;

            std::lock_guard<std::mutex> lk(registryMutex);
            auto& entry = registry[deviceId];
            auto scheduler = entry.lock();
            if(!scheduler)
            {
                scheduler = std::make_shared<OmpTaskScheduler>();
                entry = scheduler;
            }
            return scheduler;
        }

        std::future<void> submit(Operation op)
        {
            auto future = op.promise.get_future();
            ++op.queue->numPending;
            {
                std::lock_guard<std::mutex> lk(m_state->mutex);
                m_state->operations.push_back(std::move(op));
            }
            m_state->cond.notify_one();
            return future;
        }

    private:
        struct State
        {
            std::mutex mutex;
            std::condition_variable cond;
            std::deque<Operation> operations;
            bool stop = false;
        };

        std::shared_ptr<State> m_state;
        std::thread m_thread;

        static void generatorLoop(State& state)
        {
            while(true)
            {
                Operation* op = nullptr;
                {
                    std::unique_lock<std::mutex> lk(state.mutex);
                    state.cond.wait(lk, [&] { return state.stop || !state.operations.empty(); });
                    if(state.operations.empty())
                        break;
                    op = new Operation(std::move(state.operations.front()));
                    state.operations.pop_front();
                }
                createTask(op);
            }
            // all operations are submitted, wait until all tasks are finished
#    pragma omp taskwait
        }

        /** Create the task for an operation.
         *
         * The task takes the ownership of op.
         */
        static void createTask(Operation* op)
        {
            char* q = &op->queue->token;
            char const* w = static_cast<char const*>(op->waitToken);
            char* s = static_cast<char*>(op->signalToken);
            // some compilers do not count the usage in depend clauses
            alpaka::unused(q);

            if(w != nullptr && s != nullptr)
            {
#    pragma omp task ALPAKA_OMP_THREADSET_POOL firstprivate(op) depend(inout : q[0]) depend(in : w[0])                \
        depend(out : s[0])
                execute(op);
            }
            else if(w != nullptr)
            {
#    pragma omp task ALPAKA_OMP_THREADSET_POOL firstprivate(op) depend(inout : q[0]) depend(in : w[0])
                execute(op);
            }
            else if(s != nullptr)
            {
#    pragma omp task ALPAKA_OMP_THREADSET_POOL firstprivate(op) depend(inout : q[0]) depend(out : s[0])
                execute(op);
            }
            else
            {
#    pragma omp task ALPAKA_OMP_THREADSET_POOL firstprivate(op) depend(inout : q[0])
                execute(op);
            }
        }

        static void execute(Operation* op);
    };

    /** Task queue using the OpenMP task scheduler of a device. */
    class OmpFreeAgentTaskQueue final : public core::TaskQueue
    {
    public:
        explicit OmpFreeAgentTaskQueue(std::shared_ptr<OmpTaskScheduler> scheduler)
            : m_scheduler(std::move(scheduler))
            , m_queue(std::make_shared<OmpTaskScheduler::QueueState>())
        {
        }

        ~OmpFreeAgentTaskQueue() override
        {
            /* Wait until all operations are finished, equal to the callback thread of a non-blocking queue.
             * If the queue is destroyed from within a task we can not wait, the queue state is kept alive by the
             * pending operations.
             */
            if(!isInTask())
            {
                uint32_t numPending = m_queue->numPending.load();
                while(numPending != 0u)
                {
                    m_queue->numPending.wait(numPending);
                    numPending = m_queue->numPending.load();
                }
            }
        }

        bool isEmpty() const override
        {
            return m_queue->numPending.load() == 0u;
        }

        void const* getSchedulerId() const override
        {
            return m_scheduler.get();
        }

        //! Execute an operation in task context, used by the scheduler.
        static void execute(OmpTaskScheduler::Operation* op)
        {
            std::exception_ptr eptr;
            {
                TaskScope taskScope;
                try
                {
                    op->task->run();
                }
                catch(...)
                {
                    eptr = std::current_exception();
                }
                // destroy the task before the operation is marked as finished, equal to core::CallbackThread
                op->task.reset();
            }
            auto promise = std::move(op->promise);
            auto queue = std::move(op->queue);
            delete op;

            // the queue is empty before the future is ready
            --queue->numPending;
            queue->numPending.notify_all();
            if(eptr)
                promise.set_exception(std::move(eptr));
            else
                promise.set_value();
        }

    protected:
        std::future<void> submitTask(std::unique_ptr<Task> task, void const* waitToken, void* signalToken) override
        {
            return m_scheduler->submit(
                OmpTaskScheduler::Operation{std::move(task), std::promise<void>{}, m_queue, waitToken, signalToken});
        }

    private:
        std::shared_ptr<OmpTaskScheduler> m_scheduler;
        std::shared_ptr<OmpTaskScheduler::QueueState> m_queue;
    };

    inline void OmpTaskScheduler::execute(Operation* op)
    {
        OmpFreeAgentTaskQueue::execute(op);
    }
} // namespace alpaka::onHost::cpu

namespace alpaka::onHost::internal
{
    /** Create a non-blocking host queue which submits all operations to the OpenMP task scheduler of the device. */
    template<typename T_Platform, alpaka::concepts::QueuePolicyList T_QueuePolicyList>
    requires(T_QueuePolicyList::getQueueKind() == queueKind::ompFreeAgent)
    struct MakeQueue::Op<cpu::Device<T_Platform>, T_QueuePolicyList>
    {
        auto operator()(cpu::Device<T_Platform>& device, T_QueuePolicyList queuePolicies) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            auto queue = device.makeQueue(queuePolicies);
            queue->m_taskQueue = std::make_shared<cpu::OmpFreeAgentTaskQueue>(
                cpu::OmpTaskScheduler::get(device.getSharedPtr().get()));
            return queue;
        }
    };
} // namespace alpaka::onHost::internal

#endif
