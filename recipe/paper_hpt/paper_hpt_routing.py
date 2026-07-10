# Copyright 2026
#
# Paper-faithful synchronous batch routing (DataProto level).
#
# Reproduces mix_src/mix_trainer.py's post-reward batch reconstruction for the
# switch strategy (gamma == switch_gate == switch_gate_off): for each prompt,
# keep its n on-policy rollouts when solved (P > gamma), or replace them with a
# single SFT demonstration row when unsolved (P <= gamma). SFT-row construction
# (tokenizing the demonstration / tau into the actor's expected schema) is the
# caller's job and is passed in as `sft_rows_by_uid`, so this routing stays a
# pure DataProto transform that is unit-testable on CPU.

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from recipe.paper_hpt.paper_hpt_gate import group_success_counts, is_prompt_sft
from verl.protocol import DataProto

_HPT_IS_SFT = "hpt_is_sft"

# Tau/demonstration inputs ride along on RL rows only to build SFT rows; they are
# NOT training fields and must be stripped before concat (SFT rows do not carry
# them). Override via `aux_keys_to_drop` if the dataset names them differently.
DEFAULT_AUX_KEYS = (
    "tgt_input_ids",
    "tgt_attention_mask",
    "tgt_position_ids",
    "tgt_response_mask",
    "tau_messages",
)


def route_generated_batch(
    batch: DataProto,
    sft_rows_by_uid: dict[str, DataProto],
    *,
    gamma: float,
    success_value: float = 1.0,
    score_key: str = "token_level_scores",
    aux_keys_to_drop: tuple[str, ...] = DEFAULT_AUX_KEYS,
) -> DataProto:
    """Route a generated (reward-scored) batch into the HPT training batch.

    Args:
        batch: generated batch; must carry non_tensor `uid` (one shared id per
            prompt group) and per-token `score_key` tensor (bsz, resp_len).
        sft_rows_by_uid: for every UNSOLVED prompt uid, a 1-row DataProto holding
            the SFT demonstration already tokenized to the actor schema and marked
            ``hpt_is_sft == True`` (+ its synthetic terminal token_level_scores).
        gamma: HPT gate threshold; SFT iff P = num_correct/group_size <= gamma.
        success_value: per-row score counted as correct (binary reward => 1.0).
        score_key: batch tensor key holding token-level scores.
        aux_keys_to_drop: tau/demonstration inputs stripped from RL rows so RL and
            SFT pieces share a schema (concat requires identical keys).

    Returns:
        A new DataProto = concat over prompts of (kept RL rows | injected SFT row),
        preserving first-seen prompt order. RL rows are tagged ``hpt_is_sft=False``.

    Raises:
        ValueError: if, after alignment, the pieces do not share an identical set
            of batch / non_tensor keys (surfaces schema drift explicitly instead of
            failing deep inside DataProto.concat).
    """
    if "uid" not in batch.non_tensor_batch:
        raise KeyError("route_generated_batch requires non_tensor_batch['uid'].")
    if score_key not in batch.batch:
        raise KeyError(f"route_generated_batch requires batch['{score_key}'].")

    uids = batch.non_tensor_batch["uid"]
    scores = batch.batch[score_key].sum(dim=-1)
    counts = group_success_counts(scores, uids, success_value=success_value)

    idx_by_uid: OrderedDict[str, list[int]] = OrderedDict()
    for i, uid in enumerate(uids):
        idx_by_uid.setdefault(str(uid), []).append(i)

    pieces: list[DataProto] = []
    for uid, idxs in idx_by_uid.items():
        num_correct, group_size = counts[uid]
        if is_prompt_sft(num_correct, group_size, gamma):
            if uid not in sft_rows_by_uid:
                raise KeyError(f"unsolved prompt {uid!r} has no SFT row in sft_rows_by_uid.")
            sft_row = sft_rows_by_uid[uid]
            _require_sft_flag(sft_row, uid)
            pieces.append(sft_row)
        else:
            rl_rows = batch.select_idxs(idxs)
            _drop_keys(rl_rows, aux_keys_to_drop)
            _set_hpt_is_sft(rl_rows, False)
            pieces.append(rl_rows)

    _assert_homogeneous(pieces)
    return DataProto.concat(pieces)


_ZERO_ON_SFT = (
    "token_level_scores",
    "token_level_rewards",
    "rm_scores",
    "old_log_probs",
    "ref_log_prob",
    "rollout_log_probs",
)


def build_sft_row_from_template(template: DataProto, demo_ids, pad_id: int) -> DataProto:
    """Build one SFT row by cloning a rollout row and overwriting its response with the demo.

    Cloning guarantees the SFT row carries the SAME schema as the RL rows at routing
    time (old_log_probs, token_level_scores, …) so concat never mismatches. The prompt
    (same question) is kept; the response region is replaced by the tokenized demo and
    marked supervised. Reward/log-prob fields are zeroed (SFT rows carry no reward and
    the dual-loss SFT branch is pure NLL — advantage/old_log_prob are unused).
    """
    import copy

    from verl.utils.model import compute_position_id_with_mask

    b = template.batch
    prompts = b["prompts"]
    responses = b["responses"]
    _1, prompt_len = prompts.shape
    _2, resp_len = responses.shape
    device = responses.device

    d = min(len(demo_ids), resp_len)
    if d <= 0:
        raise ValueError("empty demo response for SFT row.")
    new_resp = torch.full((1, resp_len), int(pad_id), dtype=responses.dtype, device=device)
    new_resp[0, :d] = torch.tensor(demo_ids[:d], dtype=responses.dtype, device=device)
    resp_attn = torch.zeros((1, resp_len), dtype=torch.long, device=device)
    resp_attn[0, :d] = 1

    prompt_attn = b["attention_mask"][:, :prompt_len]
    new_attn = torch.cat([prompt_attn, resp_attn], dim=1)
    new_input = torch.cat([prompts, new_resp], dim=1)
    new_pos = compute_position_id_with_mask(new_attn)

    newb = b.clone()  # deep copy of all tensors (preserves schema)
    newb["responses"] = new_resp
    newb["input_ids"] = new_input
    newb["attention_mask"] = new_attn
    newb["position_ids"] = new_pos
    if "response_mask" in newb:
        newb["response_mask"] = resp_attn.to(newb["response_mask"].dtype)
    for k in _ZERO_ON_SFT:
        if k in newb:
            newb[k] = torch.zeros_like(newb[k])
    newb[_HPT_IS_SFT] = torch.ones(1, dtype=torch.bool, device=device)

    return DataProto(
        batch=newb,
        non_tensor_batch=copy.deepcopy(template.non_tensor_batch),
        meta_info=dict(template.meta_info),
    )


def _prompt_uids(batch: DataProto):
    if "prompt_uid" in batch.non_tensor_batch:
        return [str(x) for x in batch.non_tensor_batch["prompt_uid"]]
    if "extra_info" in batch.non_tensor_batch:
        return [str((ei or {}).get("prompt_uid")) for ei in batch.non_tensor_batch["extra_info"]]
    raise KeyError("batch has neither non_tensor 'prompt_uid' nor 'extra_info.prompt_uid'.")


def _make_pad_rows(batch: DataProto, pad: int, pad_id: int) -> DataProto:
    """Build `pad` TRUE-padding rows (clone-then-strip). The original mix_actor
    hard-DROPS its `whether_pad` rows before the loss; the modern engine's
    make_iterator cannot tolerate ragged mini-batches, so instead we make rows that
    are provably inert on every path a row can touch:

      - `response_mask = 0`      -> zero contribution to RL / SFT / entropy / ppo_kl
                                    (every term in paper_hpt_dual_loss is masked by it,
                                    numerators AND denominators), and loss_mask=0 in the
                                    engine so `batch_num_tokens` is unaffected.
      - `attention_mask` keeps EXACTLY ONE valid token (the last prompt position,
        always a real token because prompts are left-padded) -> the actor forward
        sees a length-1 sequence: ~zero wasted FLOPs, and `no_padding_2_padding`'s
        response slice becomes empty -> the re-padded log_prob row is all-zeros
        (its asserts need prompt_len > 0, hence one token rather than zero).
      - response tokens are overwritten with `pad_id` (never attended, never graded).
      - reward / log-prob fields zeroed; `hpt_is_sft=False`; unique dummy uid so the
        GRPO grouping sees singleton zero-score groups -> advantage exactly 0.
    """
    from verl.utils.model import compute_position_id_with_mask

    n = len(batch)
    extra = batch.select_idxs([i % n for i in range(pad)])
    b = extra.batch
    resp_len = b["responses"].shape[1]
    prompt_len = b["input_ids"].shape[1] - resp_len

    b["responses"] = torch.full_like(b["responses"], int(pad_id))
    input_ids = b["input_ids"].clone()
    input_ids[:, prompt_len:] = int(pad_id)
    b["input_ids"] = input_ids
    b["response_mask"] = torch.zeros_like(b["response_mask"])

    attn = torch.zeros_like(b["attention_mask"])
    attn[:, prompt_len - 1] = 1
    b["attention_mask"] = attn
    b["position_ids"] = compute_position_id_with_mask(attn).to(b["position_ids"].dtype)

    for k in _ZERO_ON_SFT + ("advantages", "returns"):
        if k in b:
            b[k] = torch.zeros_like(b[k])
    _set_hpt_is_sft(extra, False)
    extra.non_tensor_batch["uid"] = np.array([f"__pad_{n + i}__" for i in range(pad)], dtype=object)
    return extra


def _spread_indices(n_real: int, total: int) -> list[int]:
    """Source-index order that spreads the `n_real` real rows evenly over `total`
    slots (real row j -> slot floor(j*total/n_real); pads fill the gaps).

    Why: the padded batch is CONTIGUOUSLY chunked to DP ranks and then split into
    fixed-size mini-batches. Appending pads at the end would give early steps (few
    real rows) entire pad-only ranks: their gradient is a harmless zero, but
    Metric.aggregate_dp means metric values ACROSS RANKS FIRST, so a step with 7/8
    pad-only ranks would report actor entropy/ppo_kl at 1/8 of the true value —
    poisoning entropy-floor monitoring. Even spreading keeps every rank chunk and
    every mini block at ~uniform real-row density (it also evens per-rank compute,
    the intent of the original trainer's `_balance_batch`).
    """
    slots = [(j * total) // n_real for j in range(n_real)]
    src = [-1] * total
    for j, s in enumerate(slots):
        src[s] = j
    k = 0
    for t in range(total):
        if src[t] == -1:
            src[t] = n_real + k
            k += 1
    return src


def _pad_batch_to_multiple(
    batch: DataProto, multiple: int, *, pad_id: int, spread: bool = False
) -> tuple[DataProto, int]:
    """Pad the routed batch up to a multiple of `multiple` (the GLOBAL actor
    mini-batch size) with TRUE-padding rows (see _make_pad_rows), optionally
    spreading real rows evenly across the padded batch (see _spread_indices).

    Required because the modern engine's make_iterator asserts exact divisibility
    (the paper's mix_trainer padded to a multiple of 8 and then hard-dropped the
    pads inside the actor; masking is our loss-equivalent of that drop).
    """
    n = len(batch)
    pad = (multiple - n % multiple) % multiple
    if pad == 0:
        return batch, 0
    combined = DataProto.concat([batch, _make_pad_rows(batch, pad, pad_id)])
    if spread:
        combined = combined.select_idxs(_spread_indices(n, n + pad))
    return combined, pad


def route_generated_batch_synchronous(
    batch: DataProto,
    demo_ids_by_prompt_uid: dict[str, list],
    *,
    gamma: float,
    pad_id: int,
    success_value: float = 1.0,
    score_key: str = "token_level_scores",
    pad_to_multiple: int | None = None,
    pad_spread: bool = False,
) -> tuple[DataProto, dict[str, float]]:
    """Sync HPT routing via template cloning + HPT monitoring metrics.

    For each prompt (uid): solved (P>gamma) keeps its n rollout rows for RL; unsolved
    (P<=gamma) is replaced by one SFT row cloned from its first rollout with the demo.
    An unsolved prompt whose tau is missing falls back to RL (counted) so a data gap
    never crashes training. With `pad_spread=True` the loss-neutral pad rows are
    interleaved so real rows spread evenly over DP rank chunks and mini blocks
    (metric + load balance; see _spread_indices).
    """
    if "uid" not in batch.non_tensor_batch:
        raise KeyError("route requires non_tensor_batch['uid'].")
    uids = batch.non_tensor_batch["uid"]
    puids = _prompt_uids(batch)
    scores = batch.batch[score_key].sum(dim=-1)
    counts = group_success_counts(scores, uids, success_value=success_value)

    idx_by_uid: OrderedDict[str, list[int]] = OrderedDict()
    for i, uid in enumerate(uids):
        idx_by_uid.setdefault(str(uid), []).append(i)

    pieces: list[DataProto] = []
    n_sft = n_rl = missing_tau = 0
    succ_probs, zero_groups = [], 0
    for uid, idxs in idx_by_uid.items():
        nc, gs = counts[uid]
        succ_probs.append(nc / gs)
        zero_groups += int(nc == 0)
        if is_prompt_sft(nc, gs, gamma):
            demo = demo_ids_by_prompt_uid.get(puids[idxs[0]])
            if demo is None:
                rl = batch.select_idxs(idxs)
                _set_hpt_is_sft(rl, False)
                pieces.append(rl)
                n_rl += 1
                missing_tau += 1
                continue
            pieces.append(build_sft_row_from_template(batch.select_idxs([idxs[0]]), demo, pad_id))
            n_sft += 1
        else:
            rl = batch.select_idxs(idxs)
            _set_hpt_is_sft(rl, False)
            pieces.append(rl)
            n_rl += 1

    _assert_homogeneous(pieces)
    routed = DataProto.concat(pieces)

    # Core-instrumentation contract: this fork's compute_data_metrics (metric_utils)
    # detects `hpt_is_sft` in the batch and then REQUIRES per-row non_tensor
    # `hpt_success_probability` (+ `hpt_group_uid` for the group-deduped unbiased
    # on-policy success rate). Populate them here: sp = the row's prompt-group
    # on-policy success P (constant within group; SFT rows carry their unsolved
    # group's P, gamma=0 => 0.0). Set BEFORE padding so pad clones inherit their
    # source row's group id + P and dedup collapses them into real groups --
    # zero pad distortion of hpt/onpolicy_success_rate.
    routed_uids = routed.non_tensor_batch["uid"]
    routed.non_tensor_batch["hpt_success_probability"] = np.array(
        [float(counts[str(u)][0]) / float(counts[str(u)][1]) for u in routed_uids], dtype=object
    )
    routed.non_tensor_batch["hpt_group_uid"] = np.array([str(u) for u in routed_uids], dtype=object)

    # Real-batch observability BEFORE padding: pad rows dilute the core trainer's
    # row-mean data metrics (critic/score/*, response_length/*), so emit exact
    # pre-pad versions under hpt/ (these are the authoritative ones to monitor).
    real_resp_tokens = routed.batch["response_mask"].sum(dim=-1).float()
    real_scores = routed.batch[score_key].sum(dim=-1).float()
    n_real_rows = len(routed)

    pad_rows = 0
    if pad_to_multiple:
        routed, pad_rows = _pad_batch_to_multiple(
            routed, int(pad_to_multiple), pad_id=int(pad_id), spread=pad_spread
        )
    n_groups = max(1, n_sft + n_rl)
    metrics = {
        "hpt/num_sft": float(n_sft),
        "hpt/num_rl_groups": float(n_rl),
        "hpt/offline_data_ratio": float(n_sft) / n_groups,
        "hpt/onpolicy_success_rate": float(sum(succ_probs) / max(1, len(succ_probs))),
        "hpt/p_success_zero_ratio": float(zero_groups) / max(1, len(succ_probs)),
        "hpt/missing_tau_count": float(missing_tau),
        "hpt/pad_rows": float(pad_rows),
        "hpt/real_rows": float(n_real_rows),
        "hpt/real_response_length_mean": float(real_resp_tokens.mean().item()),
        "hpt/real_score_mean": float(real_scores.mean().item()),
    }
    return routed, metrics


def _drop_keys(batch: DataProto, keys) -> None:
    present_batch = [k for k in keys if k in batch.batch]
    present_nt = [k for k in keys if k in batch.non_tensor_batch]
    if present_batch or present_nt:
        batch.pop(batch_keys=present_batch, non_tensor_batch_keys=present_nt)


def _assert_homogeneous(pieces: list[DataProto]) -> None:
    if not pieces:
        raise ValueError("route_generated_batch produced no pieces (empty batch?).")
    ref_b = set(pieces[0].batch.keys())
    ref_n = set(pieces[0].non_tensor_batch.keys())
    for i, p in enumerate(pieces[1:], start=1):
        b, n = set(p.batch.keys()), set(p.non_tensor_batch.keys())
        if b != ref_b or n != ref_n:
            raise ValueError(
                "route_generated_batch: RL and SFT pieces have mismatched schemas "
                "(concat requires identical keys). "
                f"piece0 batch={sorted(ref_b)} non_tensor={sorted(ref_n)}; "
                f"piece{i} batch={sorted(b)} non_tensor={sorted(n)}; "
                f"batch diff={sorted(ref_b ^ b)} non_tensor diff={sorted(ref_n ^ n)}. "
                "Ensure build_sft_rows_by_uid emits the same training keys as RL rows "
                "(and that all tau/aux inputs are in aux_keys_to_drop)."
            )


def _set_hpt_is_sft(batch: DataProto, value: bool) -> None:
    n = len(batch)
    batch.batch[_HPT_IS_SFT] = torch.full((n,), bool(value), dtype=torch.bool)


def _require_sft_flag(sft_row: DataProto, uid: str) -> None:
    if _HPT_IS_SFT not in sft_row.batch:
        raise KeyError(f"SFT row for {uid!r} must carry batch['{_HPT_IS_SFT}'].")
    if not bool(sft_row.batch[_HPT_IS_SFT].all()):
        raise ValueError(f"SFT row for {uid!r} must have hpt_is_sft == True on all rows.")
