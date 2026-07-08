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
keeping informative-but-slightly-stale groups. These tests pin:
  (1) the producer policy ``zero_variance_evict_hint`` (only k==n RL groups hinted; SFT never),
  (2) the transport selection ``first_hinted_index`` (oldest hinted first; None -> oldest-first),
so the training math is untouched (only queue scheduling changes).
"""

from collections import deque

import pytest

from verl.experimental.fully_async_policy.hpt_gate import (
    HptRouteMetadata,
    zero_variance_evict_hint,
)
from verl.experimental.fully_async_policy.message_queue import first_hinted_index


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
