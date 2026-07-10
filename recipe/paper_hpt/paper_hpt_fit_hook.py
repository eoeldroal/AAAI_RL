# Copyright 2026
#
# Driver-side HPT routing entrypoint called from RayPPOTrainer.fit() (gated,
# default-off). Keeps the core hook to 3 lines: all logic lives here.
#
#   fit():  ... reward -> [route_in_fit] -> compute_advantage -> update_actor
#
# Lazily builds the prompt_uid -> demo lookup once (from the train parquet + the
# trainer tokenizer), then routes the reward-scored batch: solved prompts keep
# their rollouts for RL, unsolved prompts are replaced by one SFT demonstration row.
# Also injects `paper_hpt_beta` into the batch meta so paper_hpt_dual_loss reads it.

from __future__ import annotations

import math
from typing import Any

from recipe.paper_hpt.paper_hpt_routing import route_generated_batch_synchronous
from recipe.paper_hpt.paper_hpt_tau import load_demo_response_ids

PAPER_HPT_BETA_KEY = "paper_hpt_beta"


def _first_train_file(train_files) -> str:
    if isinstance(train_files, str):
        return train_files
    try:
        return str(train_files[0])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not resolve a train parquet from data.train_files={train_files!r}") from exc


def route_in_fit(trainer: Any, batch: Any):
    """Route the reward-scored batch for HPT; returns (routed_batch, metrics)."""
    cfg = trainer.config
    hpt = cfg.algorithm.get("paper_hpt", {})

    if getattr(trainer, "_paper_hpt_demos", None) is None:
        trainer._paper_hpt_demos = load_demo_response_ids(
            _first_train_file(cfg.data.train_files),
            trainer.tokenizer,
            max_response_length=int(cfg.data.max_response_length),
        )

    pad_id = trainer.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = trainer.tokenizer.eos_token_id

    # Padding divisor for the variable-size routed batch. It is NOT enough to be
    # divisible by the DP world size: _update_actor (ray_trainer) sets the actor
    # mini_batch_size = ppo_mini_batch_size * rollout.n (GLOBAL), and train_mini_batch
    # then requires each DP rank's slice (total // dp_size) to be divisible by
    # mini_batch_size // dp_size -> i.e. the routed TOTAL must be divisible by the global
    # mini-batch size. HPT routing shrinks the batch (unsolved: n rollouts -> 1 SFT row),
    # so a raw routed size is generally not a multiple of it (crash: "N % M != 0" in
    # make_iterator). Pad up to lcm(global_mini_batch, world) so BOTH the mini-batch
    # split and the DP dispatch divide cleanly. Pad rows are loss-neutral.
    world = int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.get("nnodes", 1))
    global_mini_batch = int(cfg.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
        cfg.actor_rollout_ref.rollout.n
    )
    pad_multiple = global_mini_batch * world // math.gcd(global_mini_batch, world)

    routed, metrics = route_generated_batch_synchronous(
        batch,
        trainer._paper_hpt_demos,
        gamma=float(hpt.get("gamma", 0.0)),
        success_value=float(hpt.get("success_value", 1.0)),
        pad_id=int(pad_id),
        pad_to_multiple=pad_multiple,
        # Spread real rows evenly over the padded batch: contiguous DP chunking would
        # otherwise give early steps pad-only ranks, whose zero metrics are averaged
        # ACROSS RANKS by Metric.aggregate_dp (entropy/ppo_kl readings collapse toward
        # real_fraction x true). Also evens per-rank compute (paper's _balance_batch intent).
        pad_spread=True,
    )
    # dual-loss reads beta from the batch meta (default already 0.3, so safe even if
    # meta propagation ever changes).
    routed.meta_info[PAPER_HPT_BETA_KEY] = float(hpt.get("beta", 0.3))
    return routed, metrics
