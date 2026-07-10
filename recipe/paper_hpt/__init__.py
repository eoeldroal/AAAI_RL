# Copyright 2026
#
# Paper-faithful synchronous HPT reproduction — ISOLATED recipe.
#
# Reproduces the ORIGINAL (Tsinghua UPT/HPT, arXiv:2509.04419) *synchronous*
# Hybrid Post-Training algorithm on the modern verl stack, kept separate from the
# fork's fully-async HPT path (`verl/experimental/fully_async_policy/`):
#   - the explicit paper dual-loss lives in `paper_hpt_loss.paper_hpt_dual_loss`,
#     wired via the opt-in `actor.custom_loss_fn` hook (engine_workers) — the only
#     shared-tree change, default-off;
#   - the gate/routing are pure driver-side logic (`paper_hpt_gate`, `paper_hpt_routing`);
#   - the trainer is a thin RayPPOTrainer subclass (`paper_hpt_trainer`).
#
# See README.md for the 3-way (paper-text / paper-code / this-recipe) mapping.
