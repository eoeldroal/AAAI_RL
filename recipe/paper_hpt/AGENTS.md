# recipe/paper_hpt — Paper-faithful SYNCHRONOUS HPT (agent guide)

Isolated reproduction of the **original UPT/HPT** (Tsinghua, arXiv:2509.04419) *synchronous*
Hybrid Post-Training algorithm on modern verl / 8×B200, to compare against this fork's
fully-async HPT (`verl/experimental/fully_async_policy/`). This file is the single source of
truth for agents working here — read it before editing.

Reproduction target = the paper's **code** (which produced Table 3), not the paper's text
(text eq.11 was never run and is internally inconsistent — clip/std vs the code's no-clip/no-std).

---

## ★ The only two shared-tree touches (both gated, default-off)

Everything else lives under `recipe/paper_hpt/`. Do NOT add more core edits; these two are
no-ops unless the paper run turns them on, so non-HPT / async runs are byte-for-byte unchanged.

1. **custom actor-loss hook** — spans TWO files (one mechanism):
   - `verl/workers/engine_workers.py` — `if actor.custom_loss_fn: use it` (else `ppo_loss`).
   - `verl/workers/config/actor.py` — `ActorConfig.custom_loss_fn: Optional[str] = None` field
     declaration. **REQUIRED**: without it, `omega_conf_to_dataclass(config.actor)` raises
     `TypeError: ... unexpected keyword argument 'custom_loss_fn'` — the reader's `.get()` is
     safe, but Hydra rejects the undeclared kwarg at *instantiation*. Keep the pair together.
   The paper run sets `custom_loss_fn=recipe.paper_hpt.paper_hpt_loss.paper_hpt_dual_loss`.
2. **`verl/trainer/ppo/ray_trainer.py` `fit()`** — after reward, before `compute_advantage`:
   `if algorithm.paper_hpt.enable: batch, m = route_in_fit(self, batch)`.

If you refactor any of these core files, preserve the hooks (gated + default-off). NB:
`algorithm.paper_hpt.*` needs NO dataclass field — `algorithm` stays an OmegaConf DictConfig read
via `.get("paper_hpt", {})`; only the `actor` worker config is converted to a strict dataclass.

---

## Paper-faithful design (preserve these — they ARE the reproduction)

- **Explicit dual-loss** (`paper_hpt_loss.paper_hpt_dual_loss`), NOT the async reward-injection:
  `L = sum_RL(-A·ratio)/L_const  +  β·mean_SFT(-logπ)  −  entropy_coeff·mean_all(entropy)`.
  RL = no-clip, sum / constant `loss_scale_factor` (=8192). SFT = pooled `masked_mean` (uniform per
  token). Entropy over ALL rows incl. demo. Pure math is `paper_hpt_dual_loss_core` (CPU-tested vs
  the paper's literal `mix_core_alg` formulas). ***Why not reward-injection:*** under
  seq-mean-token-sum-norm it does NOT equal `β·masked_mean` (length_inverse over-weights short rows;
  constant is off by T/(gbs·L)) — numerically verified.
- **Gradient-scale contract (speed-comparison critical): NO `dp_size` multiplier.** Both the paper's
  FSDP1 run (grad_accum=1, per-micro `loss.backward()` unscaled) and the modern engine (plain
  per-micro backward, FSDP2 mean-reduce) give per-step gradient `(1/R)·Σ_{ranks,micros} ∇L_local` —
  identical iff the loss is the LOCAL per-micro value. The shared `ppo_loss` multiplies by dp_size
  only because its denominator is the GLOBAL `batch_num_tokens`; ours are local like the paper's.
  A dp_size multiplier makes gradients R× the paper's — invisible to Adam but NOT to grad-clip@1.0
  (fires at 1/R threshold → systematically smaller steps → learning-speed comparison invalid).
  Pinned by `test_no_dp_size_multiplier_contract`. Optimizer parity is otherwise exact: AdamW
  (lr 5e-6, betas (0.9,0.999), wd 0.01, eps 1e-8) both sides, constant schedule / warmup 0, clip 1.0
  at the same gradient scale, `ceil(routed/512)` optimizer steps per iteration in both. Known
  as-run residual: the paper additionally SKIPS a step when pre-clip norm > 80 (`max_grad_norm`);
  modern engine skips only non-finite norms — rare event, difference bounded by one norm-≤1.0 step
  (the async arm made the same choice, M7 "[B0]").
- **Grader = `entropy_math`** (paper), via `verl/utils/reward_score/upt_v6_adapter.py` which runs it
  in a spawn subprocess (SIGALRM works there; timeout/crash → 0.0, never raises). `entropy_math` is
  **boxing-required** (no `\boxed{}` ⇒ 0) — this is the paper's intended boxing incentive, NOT a bug.
  `entropy_math_path` points at the `Unify-Post-Training` clone.
- **Data = STRIP** (`datas/openr1_hpt_main`, `keep_system_prompt=false`), NOT `_v2`. Reproduces the
  paper's system-prompt strip (low base MATH-500 ~32.8, format-collapse). Val = 6 benchmark parquets
  under the same dir (top-level `test.parquet` is EMPTY — never use it).
- **Batch = 128 prompt groups**, **grain `ppo_mini_batch_size=64`** (paper; async uses 32), Dr.GRPO
  (`norm_adv_by_std_in_grpo=false`), γ=0.0, β=0.3, entropy 0.001, KL off, temp 1.0 / n=8, lr 5e-6,
  **grad-clip 1.0** (paper yaml `grad_clip: 1.0`; train.sh's `MAX_GRAD_NORM=80` is a separate
  SKIP-the-step threshold on the PRE-clip norm, NOT the clip value — do not set clip_grad=80. The
  async arm made the same 1.0 parity choice, M7 "[B0]". The >80 step-skip itself is unimplemented:
  rare, and with clip@1.0 the difference is one norm-≤1.0 step vs none), `ppo_epochs=1`.
- **SFT rows via template cloning** (`build_sft_row_from_template`): clone one rollout row of an
  unsolved prompt, overwrite its response with the tokenized demo (tau). This auto-matches the RL-row
  schema — do NOT reconstruct SFT rows from scratch.

**vs async (what legitimately differs):** objective (dual-loss / no-clip / masked-mean SFT /
entropy-all / grain 64), architecture (sync colocated 8-GPU vs async disaggregated 6+2), data
(strip vs v2), grader (entropy_math vs math_verify). Everything else matches async.

---

## File map

| file | role |
|---|---|
| `paper_hpt_loss.py` | `paper_hpt_dual_loss` (+ `_core`): the explicit RL+SFT+entropy loss. Wired via `custom_loss_fn`. |
| `paper_hpt_gate.py` | `group_success_counts`, `is_prompt_sft` (eq.10 gate, P≤γ). |
| `paper_hpt_routing.py` | `route_generated_batch_synchronous` (gate + template cloning + HPT metrics + **DP-divisor loss-neutral padding**), `build_sft_row_from_template`. |
| `paper_hpt_tau.py` | `load_demo_response_ids`: prompt_uid → demo response tokens (assistant turn + EOS) from the train parquet. |
| `paper_hpt_fit_hook.py` | `route_in_fit(trainer, batch)`: lazy-loads tau, calls routing, injects `paper_hpt_beta` meta, passes DP world size. Called by the core fit hook. |
| `config/paper_hpt_qwen25_math_1_5b.yaml` | reference config knobs. |
| `run_paper_hpt_qwen25_math_1_5b.sh` | **main-run launcher** (entry `main_ppo trainer.use_v1=false` → v0 `RayPPOTrainer`). |
| `run_phase0_smoke_grpo.sh` | Phase-0 smoke: vanilla sync GRPO (no HPT), 20 steps — used to de-risk the sync path. |
| `tests/*_cpu.py` | 44 CPU contract tests (run in the RL conda env). |
| `paper_hpt_trainer.py` | ⚠ **LEGACY/superseded** by the fit-hook design (subclass + tgt_* passthrough); kept as dead code. The live path is the fit hook, not `PaperHptTrainer`. Safe to delete with its test. |

---

## fit() data flow (where routing sits)

`generate → old_log_prob → reward (token_level_scores) → [paper_hpt route_in_fit] → compute_advantage → update_actor(dual-loss)`

At the hook, RL rows already carry `old_log_probs` + `token_level_scores` (+ all gen fields), so
cloning gives SFT rows the full schema. Routing: solved prompt (P>γ) keeps its n rollouts (RL);
unsolved (P≤γ) → 1 SFT row from tau; rows tagged `hpt_is_sft`. GRPO advantage groups by `uid`
(singleton SFT groups → adv 0, unused). The dual-loss reads `hpt_is_sft` to split RL/SFT.

---

## Invariants an agent MUST keep (each prevents a real crash/incorrectness)

1. **TRUE-padding to the mini-batch divisor** (`pad_to_multiple` = `lcm(ppo_mini_batch_size ×
   rollout.n, world)`, `pad_spread=True`): the routed batch is variable-size and HPT SHRINKS it
   (unsolved: n rollouts → 1 SFT row). It is NOT enough to be divisible by the DP world size:
   `_update_actor` sets the actor `mini_batch_size = ppo_mini_batch_size × rollout.n` (GLOBAL), and
   `train_mini_batch` asserts each rank's slice is divisible by `mini_batch_size // dp_size` → the
   routed TOTAL must be a multiple of the global mini-batch (else `make_iterator`: "N % M != 0").
   The original mix_actor hard-DROPPED its `whether_pad` rows before the loss; the modern engine
   can't take ragged minis, so our pads are engineered to be EXACTLY drop-equivalent
   (`test_paper_hpt_padding_cpu.py` proves bit-invariance even with pathological pad values):
   - `response_mask=0` ⇒ zero contribution to RL / SFT / entropy / ppo_kl, numerators AND
     denominators (clamp_min guards make all-pad micros finite zeros); engine `loss_mask` /
     `batch_num_tokens` likewise unaffected.
   - `attention_mask` keeps exactly ONE valid token (last prompt position) ⇒ ~zero wasted FLOPs and
     an empty response slice in `no_padding_2_padding` (its asserts need prompt_len ≥ 1 — never zero
     the whole row). Response tokens overwritten with pad_id.
   - dummy uid ⇒ singleton zero-score GRPO group ⇒ advantage exactly 0; reward/log-prob fields zeroed.
   - **pad_spread** interleaves real rows evenly (real row j → slot ⌊j·total/n_real⌋): contiguous DP
     chunking + `Metric.aggregate_dp` (means ACROSS RANKS first) would otherwise collapse loss-time
     entropy/ppo_kl readings to ~real_fraction × true when early steps leave whole ranks pad-only;
     spreading also evens per-rank compute (the original `_balance_batch`'s intent).
   Don't lower the divisor back to n_gpus; don't disable the spread.
2. **GRPO advantage uses `index=uid`** — handles variable groups + singletons. Don't switch to a
   num_repeat reshape.
3. **`hpt_is_sft` is required by the dual-loss** and is only set when routing runs (training update;
   NOT validation, which has no actor loss). Keep the loss + the routing hook enabled together.
4. **`paper_hpt_beta`** flows via batch meta; default in the loss is 0.3 (= our value), so it is safe
   even if meta propagation changes.
5. **`prompt_uid`** must reach the batch (RLHFDataset carries it + `extra_info` fallback). Missing tau
   ⇒ that prompt falls back to RL (counted `hpt/missing_tau_count`), never crashes.
6. Keep RL=sum/L and SFT=mean(÷T) **asymmetry** (the paper's; async unified both to sum-norm).
7. **Core-instrumentation contract**: this fork's `compute_data_metrics` (metric_utils.py:486)
   detects `hpt_is_sft` in any batch and then REQUIRES per-row non_tensor
   `hpt_success_probability` (+ `hpt_group_uid`), crashing otherwise. The routing populates them
   (sp = the row's group on-policy success P; pads inherit their clone source's group id + P so the
   group-deduped `hpt/onpolicy_success_rate` is exactly pad-free). All other core `hpt_` reactors
   are inert here: `hpt_generated_response_lengths` (never set → early return), rollout_corr
   (config off + runs pre-routing), `losses.py` HPT path (replaced by `custom_loss_fn`). Note
   pads count as "aborted" rows (response_length 0) → `response_length/aborted_ratio` reads
   ≈ pad fraction by construction; score/reward/adv means EXCLUDE aborted rows, so they stay clean.

---

## Run

```bash
cd /NHNHOME/WORKSPACE/RL/verl
LOG=logs/paper_hpt_sync_$(date +%Y%m%d_%H%M%S).log
nohup bash recipe/paper_hpt/run_paper_hpt_qwen25_math_1_5b.sh > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"; tail -f "$LOG"
```
- Entry = `verl.trainer.main_ppo trainer.use_v1=false` (V1 is the DEFAULT; the recipe targets the
  **v0** `RayPPOTrainer` reached via `use_v1=false`).
- wandb: project `async-hpt-openr1` (same as async, for comparison), distinct `experiment_name`.
- Dumps: `trainer.rollout_data_dir` + `trainer.validation_data_dir` (both on; GBs of disk).
- HPT metrics on wandb: `hpt/offline_data_ratio` (= paper Fig 6), `num_sft`, `onpolicy_success_rate`,
  `p_success_zero_ratio`, `missing_tau_count`, `pad_rows`, `real_rows`,
  `real_response_length_mean`, `real_score_mean` (pre-pad, authoritative — the core trainer's
  `critic/score/*` and `response_length/*` row-means are diluted by pad rows).
- **Entropy monitoring**: the headline `actor/entropy` (logged from the old_log_prob pass) is
  computed PRE-routing on real rollout rows — already pad-free; use it as the primary signal. The
  loss-time MEAN metrics are near-true after pad spreading; for exact update-time values use the
  dilution-proof SUM pairs: `hpt/entropy_sum ÷ hpt/response_token_count`,
  `hpt/sft_nll_sum ÷ hpt/sft_response_token_count`, `hpt/ppo_kl_sum ÷ hpt/rl_response_token_count`
  (ratios of SUM metrics are exact under the engine's rank-mean-then-micro-sum reduction).

## Test

```bash
CUDA_VISIBLE_DEVICES="" /home/sogang_nlpy/miniconda3/envs/RL/bin/python -m pytest -q recipe/paper_hpt/tests/
```
Use the **RL** conda env (`envs/RL`, has flash_attn/sglang; read-only — do not install). 44 tests.

---

## Status & residual risks

- **CPU: fully verified** (44 tests) — dual-loss vs paper formulas, gate, routing + template cloning,
  DP padding, tau loading, real `compute_grpo_outcome_advantage`.
- **Phase-0 smoke: PASSED** — vanilla sync GRPO ran clean on B200 (v0 RayPPOTrainer + colocated
  sglang), rollout↔actor logprob pearson >0.999 (numerically consistent), val accuracies sane.
- **Phase-1 (HPT wiring): first launch shook out 3 GPU-only bugs (all fixed):**
  1. `actor.custom_loss_fn` was read but not declared → added the field to `ActorConfig`
     (`omega_conf_to_dataclass` rejected the undeclared kwarg).
  2. reward crashed `signal only works in main thread` — the async reward_loop runs `compute_score`
     in a ThreadPoolExecutor and `entropy_math` uses SIGALRM. Fix = launcher now passes
     `reward.custom_reward_function.reward_kwargs.use_process_pool=True` (+`process_timeout=30.0`) so
     the scorer runs in a spawn subprocess (its main thread). NB the reward is computed INLINE in the
     rollout manager's reward_loop, then read back via `rm_scores` → `token_level_scores` (ray_trainer
     ~1596), which the routing hook (~1628) reads. Grader runs once; no driver-side recompute.
  3. `make_iterator: N % M != 0` — see invariant #1 (pad to the global mini-batch, not n_gpus).
- **Reward throughput** is the remaining perf watch (single-worker spawn pool per RewardLoopWorker);
  correctness is safe (timeout/dead worker → 0.0). Not a blocker.
- **Baseline run LIVE: wandb `v96fvd0p`** (2026-07-09 20:42, user decision: run to completion).
  AS-RUN DELTA vs current disk code: the run imported the PRE-fix loss WITH the `×dp_size`
  multiplier (8× gradient scale) under clip_grad=80 ⇒ effective clip = 10 at paper scale, binding
  only on rare spike steps. Constant scale ⇒ Adam-neutral on normal steps; trajectory internally
  consistent and learning. Consequences:
  1. Its logged `actor/grad_norm` / `actor/pg_loss` / `rl_loss` / `sft_loss` are **8× paper scale**
     — never quote raw; val / success-rate / length metrics are unaffected.
  2. **NEVER crash-resume this run with the current (fixed) loss code**: gradient scale would flip
     8×→1× mid-run against stale Adam second moments (β₂=0.999 ⇒ hundreds of perturbed steps).
     If it dies: restart fresh, or temporarily restore the `×dp_size` line for the resume leg.
  Any NEW launch uses the fixed local-scale loss + clip 80 (user-pinned; both deviate from the
  paper's as-run clip@1.0+skip@80 by decision — M5's clip 80 is likewise inert at its own scale).
- **GPU-verified live through step 40+**: 1-token pads ≈ free (update token base ~1.13M ≈ real rows
  only), routing, dual-loss, instrumentation fields, process-pool reward all clean.
- **Speed-comparison metric semantics** (for the paper): sync wandb step = 1 trainer step = exactly
  128 groups; the async arm's wandb step = one param-sync cycle = 4 fit-steps × 128 groups, with
  `hpt/onpolicy_num_groups` and `timing_s/*` SUM-aggregated over the cycle
  (`fully_async_policy/detach_utils.py` aggregation rules) — each group counted once when trained,
  so cumulative group counts are exact on both arms. Valid cross-arm time axis: true wall-clock
  (`_runtime`) or cumulative consumed groups; never raw step indices.
- Early training on Qwen2.5-Math-1.5B base has MANY unsolved prompts → small routed batches padded up
  to 512 (loss-neutral). Expect 40–75% pad rows early = wasted actor forward compute, not wrong grads.
- **Baseline run LIVE: wandb `v96fvd0p`** (launched 2026-07-09 20:42; decision: run to completion).
  AS-RUN DELTA vs current disk code: it imported the PRE-fix loss WITH the `×dp_size` multiplier
  (8× gradient scale) under clip_grad=80 ⇒ effective clip = 10 at paper scale, binding only on rare
  spike steps (2 of first 40). The 8× is constant for the whole run ⇒ Adam-neutral on normal steps;
  trajectory is internally consistent and learning (val rising every bench, sft_loss falling).
  Consequences:
  1. This run's logged `actor/grad_norm` / `pg_loss` / `rl_loss` / `sft_loss` are **8× paper scale**
     — never quote raw values in the paper; val / success-rate / length metrics are unaffected.
  2. **NEVER crash-resume this run under the current (fixed) loss code**: gradient scale would flip
     8×→1× mid-run against stale Adam second moments (β₂=0.999 ⇒ hundreds of perturbed steps).
     If it dies: restart fresh, or temporarily restore the `×dp_size` line just for the resume.
  Any NEW launch runs the fixed local-scale loss + clip 80 (user-pinned; deviates from the paper's
  as-run clip@1.0 + skip@80 by decision — M5's clip 80 is likewise inert at its own loss scale).
- **GPU-verified live through step 40+**: 1-token pads ≈ free (update token base ~1.13M/step ≈ real
  rows only — the padding design confirmed empirically), routing, dual-loss, instrumentation fields,
  process-pool reward all clean; step ≈ 39s (gen 22s / logprob 4s / update 8s).
