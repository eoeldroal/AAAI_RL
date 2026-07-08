# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Contract tests for B1' semantic-aware queue eviction (Improvement_RL.md §5.12).

Fully-async production overproduces relative to trainer consumption (M5 logs: ~7940 queue
drops; ~165 groups/step discarded), so the completed-groups queue is CONSTANTLY at capacity
and dropping. Which entry to drop is a free design choice: the stock policy drops the oldest
(``popleft``); B1' instead drops producer-marked zero-information all-correct (k==n) RL groups
first, since their GRPO advantage is identically zero (dead weight in the training batch),
keeping informative-but-slightly-stale groups. These tests pin three layers:
  (1) the producer policy ``zero_variance_evict_hint`` (only k==n RL groups hinted; SFT never),
  (2) the transport selection ``first_hinted_index`` (oldest hinted first; None -> oldest-first),
  (3) the MessageQueue actor's put/get behavior itself (undecorated class, no Ray cluster):
      the (hint, payload) tuple is queue-internal (consumers receive payloads unchanged),
      overflow victim priority is incoming-hinted -> queued-hinted -> oldest, and the
      dropped/evicted counters stay consistent,
so the training math is untouched (only queue scheduling changes).
"""

import asyncio
from collections import deque

import pytest

from verl.experimental.fully_async_policy.hpt_gate import (
    HptRouteMetadata,
    zero_variance_evict_hint,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueue, first_hinted_index

# The undecorated actor class: same pattern as test_hpt_trainer_queue_contract's use of
# FullyAsyncRollouter.__ray_metadata__.modified_class — drives the REAL put/get code paths
# in-process without a Ray cluster.
_QueueCls = MessageQueue.__ray_metadata__.modified_class


def _route(*, is_sft: bool, success_count: int, total_count: int = 8) -> HptRouteMetadata:
    return HptRouteMetadata(
        is_sft=is_sft,
        prompt_uid="p",
        group_uid="g",
        success_probability=success_count / total_count,
        success_count=success_count,
        total_count=total_count,
        gamma=0.0,
        success_threshold=0.0,
        success_score_key="reward_score",
    )


# ---- producer policy: zero_variance_evict_hint ----


def test_all_correct_rl_group_is_hinted_only_when_enabled():
    all_correct = _route(is_sft=False, success_count=8)
    assert zero_variance_evict_hint(all_correct, enabled=True) == 1
    # Feature off -> never hint (exact original age-only policy preserved).
    assert zero_variance_evict_hint(all_correct, enabled=False) == 0


def test_informative_groups_are_never_hinted():
    # k in 1..n-1 carry nonzero GRPO variance -> keep.
    for k in range(1, 8):
        assert zero_variance_evict_hint(_route(is_sft=False, success_count=k), enabled=True) == 0
    # k==0 is routed to teacher-SFT (is_sft), not an RL zero-variance group -> keep.
    assert zero_variance_evict_hint(_route(is_sft=True, success_count=0), enabled=True) == 0


def test_sft_group_is_never_hinted_preserving_rl_sft_mix():
    # Even a degenerate all-"correct" SFT metadata must not be hinted: dropping teacher rows
    # would distort the RL/SFT consumption ratio (the failure mode UPT Table 6 warns against).
    assert zero_variance_evict_hint(_route(is_sft=True, success_count=8), enabled=True) == 0


def test_none_route_is_safe():
    # Non-HPT samples carry no route metadata -> hint 0, no crash.
    assert zero_variance_evict_hint(None, enabled=True) == 0
    assert zero_variance_evict_hint(None, enabled=False) == 0


# ---- transport selection: first_hinted_index ----


def test_no_hint_falls_back_to_oldest_first():
    # All hints 0 -> None -> caller does popleft (original behavior, bit-for-bit).
    q = deque([(0, "a"), (0, "b"), (0, "c")])
    assert first_hinted_index(q) is None


def test_oldest_hinted_entry_is_selected():
    # Two hinted entries; select the OLDEST (lowest index) so age still breaks ties among victims.
    q = deque([(0, "keep0"), (1, "old_k8"), (0, "keep1"), (1, "new_k8")])
    assert first_hinted_index(q) == 1


def test_empty_queue_returns_none():
    assert first_hinted_index(deque()) is None


def test_eviction_preserves_informative_over_stale_end_to_end():
    # Simulate the overflow decision the actor makes: a full queue of mostly-informative
    # (possibly older) groups plus one all-correct k=8 group. B1' evicts the k=8 group,
    # so every informative group -- including the oldest -- survives.
    q = deque([(0, "stale_informative"), (1, "k8_dead_weight"), (0, "fresh_informative")])
    idx = first_hinted_index(q)
    assert idx == 1
    del q[idx]
    survivors = [payload for _hint, payload in q]
    assert survivors == ["stale_informative", "fresh_informative"]
    # Without B1' (age-only), the oldest informative group would have been dropped instead.
    q2 = deque([(0, "stale_informative"), (1, "k8_dead_weight"), (0, "fresh_informative")])
    q2.popleft()  # stock policy drops the oldest...
    assert "stale_informative" not in [p for _h, p in q2]  # ...losing an informative group


# ---- MessageQueue actor behavior (undecorated class, real put/get code paths) ----


def _make_queue(max_queue_size: int):
    return _QueueCls(config=None, max_queue_size=max_queue_size)


def test_actor_roundtrip_hides_hint_tuple_from_consumer():
    # The (hint, payload) tuple is a queue-internal detail: get_sample must return the
    # payload exactly as put (plus queue length), for hinted and unhinted entries alike --
    # this is the contract that lets the trainer's collection loop stay untouched.
    async def run():
        q = _make_queue(4)
        await q.put_sample("plain", evict_hint=0)
        await q.put_sample("hinted", evict_hint=1)
        first, len_after_first = await q.get_sample()
        second, len_after_second = await q.get_sample()
        return first, len_after_first, second, len_after_second

    first, len1, second, len2 = asyncio.run(run())
    assert (first, len1) == ("plain", 1)
    assert (second, len2) == ("hinted", 0)


def test_actor_none_termination_sentinel_passes_through():
    # The rollouter signals shutdown with put_sample(sample=None). It must come back to the
    # consumer as a None payload (the trainer's termination check), not crash the hint logic.
    async def run():
        q = _make_queue(2)
        await q.put_sample(None)
        payload, _qlen = await q.get_sample()
        return payload

    assert asyncio.run(run()) is None


def test_actor_overflow_evicts_queued_hinted_before_oldest():
    # Full queue [old_informative, k8, new_informative]; an informative sample arrives.
    # The queued k8 entry must be the victim; both informative entries and the newcomer stay.
    async def run():
        q = _make_queue(3)
        await q.put_sample("old_informative")
        await q.put_sample("k8", evict_hint=1)
        await q.put_sample("new_informative")
        accepted = await q.put_sample("incoming_informative")
        stats = await q.get_statistics()
        payloads = [(await q.get_sample())[0] for _ in range(3)]
        return accepted, stats, payloads

    accepted, stats, payloads = asyncio.run(run())
    assert accepted is False  # an eviction happened on this put (unchanged return semantics)
    assert payloads == ["old_informative", "new_informative", "incoming_informative"]
    assert stats["dropped_samples"] == 1
    assert stats["evicted_hinted"] == 1


def test_actor_overflow_drops_incoming_hinted_without_displacing_informative():
    # Full queue of informative entries; a k8 (hinted) sample arrives. Enqueueing it would
    # displace an informative entry only for the k8 to be first pick on the next overflow,
    # so the incoming hinted sample itself is the victim: queue contents unchanged.
    async def run():
        q = _make_queue(2)
        await q.put_sample("info_a")
        await q.put_sample("info_b")
        accepted = await q.put_sample("k8_incoming", evict_hint=1)
        pre_drain = await q.get_statistics()
        payloads = [(await q.get_sample())[0] for _ in range(2)]
        post_drain = await q.get_statistics()
        return accepted, pre_drain, payloads, post_drain

    accepted, pre_drain, payloads, post_drain = asyncio.run(run())
    assert accepted is False
    assert pre_drain["queue_size"] == 2  # incoming k8 never entered; queue kept as-is
    assert payloads == ["info_a", "info_b"]  # informative entries untouched
    assert pre_drain["dropped_samples"] == 1
    assert pre_drain["evicted_hinted"] == 1
    assert post_drain["queue_size"] == 0  # both consumed; the k8 never lingered


def test_actor_overflow_without_hints_keeps_original_oldest_first():
    # Feature-off / non-HPT runs: no hint ever enqueued -> overflow behavior is bit-for-bit
    # the original popleft (oldest dropped), and evicted_hinted stays 0.
    async def run():
        q = _make_queue(2)
        await q.put_sample("oldest")
        await q.put_sample("middle")
        accepted = await q.put_sample("newest")
        stats = await q.get_statistics()
        payloads = [(await q.get_sample())[0] for _ in range(2)]
        return accepted, stats, payloads

    accepted, stats, payloads = asyncio.run(run())
    assert accepted is False
    assert payloads == ["middle", "newest"]
    assert stats["dropped_samples"] == 1
    assert stats["evicted_hinted"] == 0


def test_actor_counters_reconcile_under_mixed_traffic():
    # produced = accepted + displaced-on-put; consumed drains what remains; hinted evictions
    # are a subset of drops. Reconciliation guards against double counting in either branch.
    async def run():
        q = _make_queue(3)
        for i in range(3):
            await q.put_sample(f"info_{i}")
        await q.put_sample("k8_a", evict_hint=1)  # full + incoming hinted -> dropped incoming
        await q.put_sample("info_3")  # full, no hinted queued -> popleft info_0
        await q.put_sample("k8_b", evict_hint=1)  # full + incoming hinted -> dropped incoming
        stats = await q.get_statistics()
        remaining = []
        while stats["queue_size"] and (got := await q.get_sample()) is not None:
            remaining.append(got[0])
            stats = await q.get_statistics()
        return stats, remaining

    stats, remaining = asyncio.run(run())
    assert remaining == ["info_1", "info_2", "info_3"]
    assert stats["total_produced"] == 6
    assert stats["total_consumed"] == 3
    assert stats["dropped_samples"] == 3  # k8_a, info_0, k8_b
    assert stats["evicted_hinted"] == 2  # k8_a, k8_b only


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
