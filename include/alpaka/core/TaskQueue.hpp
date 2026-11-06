/* Copyright 2026 Valentin Seitz
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#include <future>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka::core
{
    /** Interface for a task based execution backend of a host queue.
     *
     * A host queue normally executes its operations in order on its own callback thread. If a task queue is attached
     * the operations are handed to the task queue instead. The implementation must keep the in-order semantics of
     * the queue, but operations of different queues are allowed to overlap.
     *
     * Dependencies between queues are expressed by dependency tokens: an operation can signal a token (written when
     * the operation finished) and another operation can wait for a token. Tokens are only valid between task queues
     * which share the same scheduler, see getSchedulerId().
     */
    class TaskQueue
    {
    public:
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
        //! \todo with C++23 std::move_only_function should be used
        struct Task
        {
            virtual ~Task() = default;
            virtual void run() = 0;
        };
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

        virtual ~TaskQueue() = default;

        /** Submit an operation.
         *
         * @param fn nullary function returning void
         * @param waitToken the operation starts not before the operation signaling this token has finished, can be
         * nullptr
         * @param signalToken token signaled when the operation has finished, can be nullptr
         * @return future which is ready after the operation has finished and is destroyed
         */
        template<typename T_Fn>
        std::future<void> submit(T_Fn&& fn, void const* waitToken = nullptr, void* signalToken = nullptr)
        {
            using DecayedFn = std::decay_t<T_Fn>;
            static_assert(
                std::is_void_v<std::invoke_result_t<DecayedFn>>,
                "Submitted function must not have any arguments and return void.");
            return submitTask(
                std::make_unique<FunctionHolder<DecayedFn>>(std::forward<T_Fn>(fn)),
                waitToken,
                signalToken);
        }

        //! @return true if no operation is pending or executed
        virtual bool isEmpty() const = 0;

        //! @return identifier of the scheduler, dependency tokens can only be used between equal identifiers
        virtual void const* getSchedulerId() const = 0;

        //! @return true if the calling thread is currently executing an operation of a task queue
        static bool isInTask()
        {
            return inTaskFlag();
        }

    protected:
        virtual std::future<void> submitTask(std::unique_ptr<Task> task, void const* waitToken, void* signalToken) = 0;

        /** Marks the calling thread as executing a task queue operation for the lifetime of the object.
         *
         * Must be used by implementations around the execution of an operation.
         */
        struct TaskScope
        {
            TaskScope() : m_previous(inTaskFlag())
            {
                inTaskFlag() = true;
            }

            ~TaskScope()
            {
                inTaskFlag() = m_previous;
            }

            TaskScope(TaskScope const&) = delete;
            TaskScope& operator=(TaskScope const&) = delete;

        private:
            bool m_previous;
        };

    private:
        template<typename T_Fn>
        struct FunctionHolder : Task
        {
            T_Fn func;

            template<typename T_FnFwd>
            explicit FunctionHolder(T_FnFwd&& funcFwd) : func{std::forward<T_FnFwd>(funcFwd)}
            {
            }

            void run() override
            {
                func();
            }
        };

        static bool& inTaskFlag()
        {
            thread_local bool flag = false;
            return flag;
        }
    };
} // namespace alpaka::core
