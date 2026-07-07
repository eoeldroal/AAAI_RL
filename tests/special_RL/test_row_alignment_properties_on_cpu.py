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
"""Property-based tests for the collection-boundary row-alignment planner.

The grow-to-align collection loop crashed on compositions whose residue mod
rollout_n could never be reached (Improvement_RL §5.8.4). ``_plan_row_alignment_deferral``
replaced it: it must, for *every* group composition, either return a deferral
subset that leaves an exactly-aligned retained batch, or return ``None`` only
when the residue is genuinely unreachable. Example-based tests pin specific
compositions; these hypothesis properties assert the invariants universally.
"""

import pytest

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

# Skip cleanly if hypothesis is not installed; access members off the module to keep
# all real imports at the top of the file (no post-statement `from` imports -> no E402).
hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = pytest.importorskip("hypothesis.strategies")

# The planner is a pure staticmethod; reach it on the undecorated Ray actor class.
_plan = FullyAsyncTrainer.__ray_metadata__.modified_class._plan_row_alignment_deferral


def _subset_sum_reachable(sizes: list[int], target: int) -> bool:
    """Brute-force 0/1 subset-sum oracle, independent of the planner's DP."""
    reachable = {0}
    for s in sizes:
        reachable |= {r + s for r in reachable if r + s <= target}
    return target in reachable


@st.composite
def _compositions(draw):
    # Group sizes are small positive ints (SFT=1, RL=rollout_n); mix them freely.
    row_counts = draw(st.lists(st.integers(min_value=1, max_value=16), min_size=0, max_size=40))
    required_multiple = draw(st.integers(min_value=1, max_value=32))
    protected_prefix = draw(st.integers(min_value=0, max_value=len(row_counts)))
    return row_counts, required_multiple, protected_prefix


@given(_compositions())
def test_returned_deferral_leaves_an_exactly_aligned_batch(params):
    row_counts, required_multiple, protected_prefix = params
    defer = _plan(row_counts, required_multiple, protected_prefix=protected_prefix)
    if defer is None:
        return  # covered by the completeness property below
    total = sum(row_counts)
    # Only eligible (non-protected / non-carried-over) groups may be deferred.
    assert all(protected_prefix <= i < len(row_counts) for i in defer)
    # Deferred rows equal exactly the residue...
    deferred_rows = sum(row_counts[i] for i in defer)
    assert deferred_rows == total % required_multiple
    # ...so the retained batch is an exact multiple (the crash-free alignment invariant).
    assert (total - deferred_rows) % required_multiple == 0


@given(_compositions())
def test_none_is_returned_exactly_when_the_residue_is_unreachable(params):
    row_counts, required_multiple, protected_prefix = params
    defer = _plan(row_counts, required_multiple, protected_prefix=protected_prefix)
    residue = sum(row_counts) % required_multiple
    eligible = row_counts[protected_prefix:]
    reachable = required_multiple <= 1 or _subset_sum_reachable(eligible, residue)
    assert (defer is not None) == reachable


@given(_compositions())
def test_aligned_or_trivial_multiple_defers_nothing(params):
    row_counts, required_multiple, protected_prefix = params
    if required_multiple <= 1 or sum(row_counts) % required_multiple == 0:
        assert _plan(row_counts, required_multiple, protected_prefix=protected_prefix) == set()
