# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def first_hinted_index(entries) -> int | None:
    """Return the index of the first (oldest) queue entry marked evict_hint>0, else None.

    Pure, Ray-free, and unit-testable. Entries are ``(evict_hint, payload)`` tuples in
    FIFO order (index 0 = oldest). Semantic-aware eviction (B1') prefers dropping these
    producer-marked zero-information entries over informative (possibly slightly stale)
    ones; None means "no marked victim -> fall back to plain oldest-first (popleft)".
    """
    for i, entry in enumerate(entries):
        if entry[0]:
            return i
    return None


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue: deque[Any] = deque(maxlen=self.max_queue_size)

        self.val_queue: deque[Any] = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0
        # B1': subset of dropped_samples that were evicted because the producer marked them
        # zero-information (evict_hint>0), rather than by plain oldest-first age.
        self.evicted_hinted = 0
        # Cheap guard: only scan for a hinted victim once any hinted entry has been enqueued,
        # so non-HPT / feature-off runs keep the exact original O(1) popleft on overflow.
        self._saw_evict_hint = False

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    def _select_evict_index(self) -> int | None:
        """Overflow victim index: first hinted entry (B1'), else None -> oldest-first.

        Delegates to the pure module-level ``first_hinted_index`` so the policy is unit
        testable without Ray. The queue stays semantics-agnostic — it only reads the int.
        """
        return first_hinted_index(self.queue)

    async def put_sample(self, sample: Any, evict_hint: int = 0) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            evict_hint: transport eviction priority (0 = normal; >0 = drop-first on overflow).
                Set by the producer for zero-information groups (see zero_variance_evict_hint).

        Returns:
            bool: Whether the sample was put without displacing another (True), or an
                overflow eviction occurred on this put (False) — unchanged semantics.
        """
        async with self._lock:
            # If queue is full, make room. Victim priority (B1'): the INCOMING sample itself
            # when it is producer-marked zero-information (enqueueing it would displace an
            # informative entry only for it to be first pick on the next overflow anyway),
            # then a marked entry already in the queue, then the oldest (original behavior).
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
                if evict_hint:
                    self.evicted_hinted += 1
                    self.total_produced += 1
                    return not is_drop
                idx = self._select_evict_index() if self._saw_evict_hint else None
                if idx is not None:
                    del self.queue[idx]
                    self.evicted_hinted += 1
                else:
                    self.queue.popleft()
            if evict_hint:
                self._saw_evict_hint = True
            self.queue.append((int(evict_hint), sample))
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            return not is_drop

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample. Entries are (evict_hint, payload) tuples; the hint is a
            # transport-internal detail, so consumers receive only the payload.
            _hint, data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "evicted_hinted": self.evicted_hinted,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = next(iter(self.queue))[1]  # entries are (evict_hint, payload) tuples
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, evict_hint: int = 0) -> bool:
        """Put batch into queue (async). evict_hint forwards the transport eviction priority."""
        future = self.queue_actor.put_sample.remote(sample, evict_hint)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
