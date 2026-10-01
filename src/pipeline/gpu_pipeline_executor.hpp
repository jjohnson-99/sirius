/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "event/query_event_publisher.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_request.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <type_traits>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::parallel {
class downgrade_executor;
}  // namespace sirius::parallel

namespace sirius::telemetry {
class telemetry_context;
}  // namespace sirius::telemetry

namespace sirius {

namespace creator {
class task_creator;
}

namespace pipeline {

//! One slot per value SiriusPhysicalOperatorType can hold, so every enumerator indexes in bounds.
inline constexpr std::size_t operator_type_slots =
  std::size_t{1} << (8 * sizeof(std::underlying_type_t<op::SiriusPhysicalOperatorType>));
static_assert(
  std::is_unsigned_v<std::underlying_type_t<op::SiriusPhysicalOperatorType>> &&
    operator_type_slots <= 256,
  "the operator-type enum must be unsigned and at most 8 bits wide to index the metric");

/**
 * @brief Snapshot of one gpu_pipeline_executor's task and cross-GPU clone counts, broken down by
 * the source operator type of each task's pipeline.
 *
 * A task whose pipeline or source is missing is counted under SiriusPhysicalOperatorType::INVALID.
 */
struct executor_metrics {
  //! Tasks whose execute() returned on this executor; the sum of tasks_by_source.
  std::size_t tasks_executed{0};
  //! The same tasks, by source operator type.
  std::array<std::size_t, operator_type_slots> tasks_by_source{};
  //! Input batches cloned from another GPU while tasks prepared, summed over attempts that
  //! completed or were rescheduled.
  std::array<std::size_t, operator_type_slots> cross_gpu_inputs_by_source{};

  //! Tasks from pipelines whose source is `type`.
  [[nodiscard]] constexpr std::size_t tasks_from(op::SiriusPhysicalOperatorType type) const noexcept
  {
    return tasks_by_source[static_cast<std::size_t>(type)];
  }

  //! Cross-GPU input clones in pipelines whose source is `type`.
  [[nodiscard]] constexpr std::size_t cross_gpu_inputs_from(
    op::SiriusPhysicalOperatorType type) const noexcept
  {
    return cross_gpu_inputs_by_source[static_cast<std::size_t>(type)];
  }
};

/**
 * @brief Executor specialized for executing GPU pipeline operations.
 *
 * This executor inherits from itask_executor and manages a pool of threads
 * dedicated to executing GPU pipeline tasks with specialized GPU resource
 * management.
 */
class gpu_pipeline_executor : public sirius::parallel::itask_executor {
 public:
  /**
   * @brief Constructs a new gpu_pipeline_executor with task execution configuration
   *
   * @param config Configuration for the task executor (thread count, retry policy, etc.)
   * @param mem_space Pointer to the memory space for GPU allocations
   * @param task_request_publisher Publisher to submit task requests
   * @param downgrade_executor Pointer to the downgrade executor. This is used so that the
   * gpu_pipeline_executor can request memory downgrade if it cannot obtain a reservation from the
   * memory space.
   */
  explicit gpu_pipeline_executor(
    exec::thread_pool_config config,
    cucascade::memory::memory_space* mem_space,
    exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
    sirius::parallel::downgrade_executor* downgrade_executor,
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context);

  /**
   * @brief Destructor for the gpu_pipeline_executor.
   */
  ~gpu_pipeline_executor();

  /// Attach the query-event observer, sharing ownership so the handle is
  /// neither null nor dangling.  Propagated by task_scheduler; until then this
  /// executor reports into its own subscriber-less publisher.
  void set_query_event_publisher(sirius::event::query_event_publisher& publisher)
  {
    _query_event_publisher = publisher.shared_from_this();
  }

  // Non-copyable but movable
  gpu_pipeline_executor(const gpu_pipeline_executor&)            = delete;
  gpu_pipeline_executor& operator=(const gpu_pipeline_executor&) = delete;
  gpu_pipeline_executor(gpu_pipeline_executor&&)                 = delete;
  gpu_pipeline_executor& operator=(gpu_pipeline_executor&&)      = delete;

  /**
   * @brief Set the task creator for scheduling output consumers
   *
   * @param task_creator Pointer to the task creator
   */
  void set_task_creator(creator::task_creator* task_creator);

  /**
   * @brief Check if the internal task queue is empty.
   *
   * Useful for verifying that drain_and_wait() has fully cleared the queue.
   * Only reliable when the executor is quiescent (no concurrent producers).
   *
   * @return true if the task queue contains no pending tasks.
   */
  [[nodiscard]] bool is_task_queue_empty() const noexcept;

  /**
   * @brief Return a snapshot of this executor's runtime metrics.
   */
  [[nodiscard]] executor_metrics get_metrics() const noexcept;

  /**
   * @brief Return the effective executor configuration after scheduler derivation.
   */
  [[nodiscard]] const exec::thread_pool_config& get_effective_config() const noexcept
  {
    return _config;
  }

 protected:
  void manager_loop() override;

  sirius::exec::invocable<void() noexcept> get_per_thread_init() override;

 private:
  /**
   * @brief Safely casts itask to gpu_pipeline_task with type validation
   *
   * @param task The itask pointer to cast
   * @return gpu_pipeline_task* The casted gpu_pipeline_task pointer
   * @throws std::bad_cast if the task is not of type gpu_pipeline_task
   */
  gpu_pipeline_task* cast_to_gpu_pipeline_task(sirius::parallel::itask* task);

  cucascade::memory::exclusive_stream_pool _stream_pool;
  exec::publisher<std::unique_ptr<task_request>> _task_request_publisher;
  cucascade::memory::memory_space* _memory_space;
  sirius::parallel::downgrade_executor* _downgrade_executor{nullptr};
  /// Observer of query event transitions.  Never null; see task_creator.
  std::shared_ptr<sirius::event::query_event_publisher> _query_event_publisher{
    std::make_shared<sirius::event::query_event_publisher>()};
  sirius::creator::task_creator* _task_creator{nullptr};
  //! Backs executor_metrics::tasks_by_source.
  std::array<std::atomic<std::size_t>, operator_type_slots> _tasks_by_source{};
  //! Backs executor_metrics::cross_gpu_inputs_by_source.
  std::array<std::atomic<std::size_t>, operator_type_slots> _cross_gpu_inputs_by_source{};
};

}  // namespace pipeline
}  // namespace sirius
