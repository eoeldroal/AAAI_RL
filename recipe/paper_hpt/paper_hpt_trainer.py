# Copyright 2026
#
# Paper-faithful synchronous HPT trainer (isolated recipe).
#
# Thin subclass of the modern verl synchronous PPO trainer that inserts ONE step
# — HPT routing — between reward computation and advantage computation, exactly
# where mix_src/mix_trainer.py does it. The paper dual-loss is wired via the
# `actor.custom_loss_fn` hook (engine_workers), so no shared loss path is altered.
#
# With the explicit dual-loss, SFT rows carry NO synthetic reward/advantage — the
# loss computes beta*masked_mean(-logpi) directly. The gate only decides SFT/RL;
# routing keeps solved prompts' rollouts and replaces unsolved prompts with one
# demonstration (tau) row marked hpt_is_sft=True.
#
# routing + gate are CPU-verified (see tests/). The GPU-bound pieces are marked
# `SMOKE:` and listed in README §Status.

from __future__ import annotations

import numpy as np
import torch

from recipe.paper_hpt.paper_hpt_routing import route_generated_batch
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

_HPT_IS_SFT = "hpt_is_sft"
PAPER_HPT_BETA_KEY = "paper_hpt_beta"


class PaperHptConfig:
    """Gate knobs for the paper reproduction (read off the hydra config)."""

    def __init__(self, algo_cfg, actor_cfg):
        hpt = algo_cfg.get("paper_hpt", {})
        self.enable: bool = bool(hpt.get("enable", False))
        self.gamma: float = float(hpt.get("gamma", 0.0))  # eq.10 gate; 0.0 for Qwen
        self.beta: float = float(hpt.get("beta", 0.3))  # SFT coefficient (1.5B); read by dual-loss
        self.success_value: float = float(hpt.get("success_value", 1.0))


def build_sft_rows_by_uid(batch: DataProto, cfg: PaperHptConfig) -> dict[str, DataProto]:
    """Materialize one SFT demonstration row per prompt uid, from the carried tau.

    With the explicit dual-loss, the SFT row needs only: the tokenized demonstration
    (input_ids/attention_mask/position_ids), a response_mask over the demonstration
    tokens, hpt_is_sft=True, and zeroed reward fields (SFT rows do NOT go through the
    advantage — the loss uses masked_mean-NLL). Assumes the dataset passes, per prompt,
    a pre-tokenized demonstration in these batch fields (see README §Data):
        tgt_input_ids / tgt_attention_mask / tgt_position_ids / tgt_response_mask.

    SMOKE: the exact key names / layout must match what the actor consumes; confirm
    against a live batch and adjust the field mapping here only.
    """
    required = ("tgt_input_ids", "tgt_attention_mask", "tgt_position_ids", "tgt_response_mask")
    missing = [k for k in required if k not in batch.batch]
    if missing:
        raise KeyError(
            f"build_sft_rows_by_uid needs tau fields {missing} in the batch; "
            "wire them through the dataset (README §Data)."
        )
    uids = batch.non_tensor_batch["uid"]
    seen: dict[str, int] = {}
    for i, uid in enumerate(uids):
        seen.setdefault(str(uid), i)

    rows: dict[str, DataProto] = {}
    for uid, i in seen.items():
        resp_mask = batch.batch["tgt_response_mask"][i : i + 1]
        zeros = torch.zeros_like(resp_mask, dtype=torch.float32)
        row = DataProto.from_single_dict(
            {
                "input_ids": batch.batch["tgt_input_ids"][i : i + 1],
                "attention_mask": batch.batch["tgt_attention_mask"][i : i + 1],
                "position_ids": batch.batch["tgt_position_ids"][i : i + 1],
                "response_mask": resp_mask,
                # SFT rows are pure NLL: no reward, no advantage, and old_log_prob is
                # unused (the RL ratio never applies to them). Zeroed for concat parity.
                "old_log_probs": zeros.clone(),
                "token_level_scores": zeros,
                "rm_scores": zeros.clone(),
                _HPT_IS_SFT: torch.ones(1, dtype=torch.bool),
                "uid": np.array([uid], dtype=object),
            }
        )
        rows[uid] = row
    return rows


class PaperHptTrainer(RayPPOTrainer):
    """Synchronous HPT via post-reward routing + the custom dual-loss.

    Integration (see README §Status): the base `fit()` computes rewards into
    `batch.batch['token_level_scores']` and then calls `compute_advantage`. Insert
    exactly one call between them:

        batch = self._paper_hpt_route(batch)   # <-- after reward, before advantage

    Because the modern base `fit()` has no post-reward hook, wiring is done by a
    minimal `fit()` override once Phase-0 smoke confirms which synchronous trainer
    generation is live on this box. Until then this method is unit-tested via its
    delegates (gate/routing) on CPU. The paper dual-loss is selected by config:
    `actor.custom_loss_fn=recipe.paper_hpt.paper_hpt_loss.paper_hpt_dual_loss`.
    """

    def _paper_hpt_cfg_or_build(self) -> PaperHptConfig:
        cfg = getattr(self, "_paper_hpt_cfg", None)
        if cfg is None:
            cfg = PaperHptConfig(self.config.algorithm, self.config.actor_rollout_ref.actor)
            self._paper_hpt_cfg = cfg
        return cfg

    def _paper_hpt_route(self, batch: DataProto) -> DataProto:
        cfg = self._paper_hpt_cfg_or_build()
        if not cfg.enable:
            return batch
        sft_rows = build_sft_rows_by_uid(batch, cfg)
        routed = route_generated_batch(
            batch,
            sft_rows,
            gamma=cfg.gamma,
            success_value=cfg.success_value,
        )
        # The dual-loss reads beta from the batch meta (not config), matching how
        # ppo_loss reads hpt flags from the batch.
        routed.meta_info[PAPER_HPT_BETA_KEY] = cfg.beta
        return routed
