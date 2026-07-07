# RL Run Notes

_Last updated: 2026-07-07_

This document records environment and launch practice for the clean
RL-focused `verl` fork. It is not a replacement for the upstream docs. For
queue/staleness/HPT budget sizing, see `AsyncBudget_RL.md`. For what this
fork is, see `Overview_RL.md`. Project-wide coding and review principles
remain in `../AGENTS.md`.

## Environment Checklist

Before a main run:

1. Check the repo state.

   ```bash
   git status --short
   git log --oneline -5
   ```

2. Confirm the environment was built from the checked-in install path and not by
   untracked manual package edits.

   ```bash
   python - <<'PY'
   import torch, sglang, sgl_kernel, flash_attn
   print("torch", torch.__version__)
   print("torch cuda", torch.version.cuda)
   print("sglang", sglang.__version__)
   print("sglang-kernel", sgl_kernel.__version__)
   print("flash-attn", flash_attn.__version__)
   print("cuda available", torch.cuda.is_available())
   PY
   ```

   The RL environment exposes CUDA 13, cuDNN, and NCCL wheel libraries through
   its conda activation hook. A run that only prepends the RL environment's
   `bin` directory to `PATH` is not equivalent to `conda activate RL`: SGLang
   subprocesses can fail during scheduler initialization with
   `libcudart.so.13: cannot open shared object file`. Main launchers should
   activate the RL conda environment themselves or be run from a freshly
   activated RL shell.

   The cluster `/tmp` can be mounted `noexec`. Triton and TorchInductor build
   small shared objects at runtime, so leaving their cache under `/tmp` can make
   actor update fail with:

   ```text
   __triton_launcher...so: failed to map segment from shared object
   ```

   Main launchers should keep JIT artifacts on an executable project filesystem.
   Keep the path short because Ray also creates AF_UNIX sockets under `TMPDIR`
   and Linux socket paths are limited to about 107 bytes:

   ```text
   TMPDIR=<short executable path>/tmp
   TRITON_CACHE_DIR=<short executable path>/triton
   TORCHINDUCTOR_CACHE_DIR=<short executable path>/torchinductor
   XDG_CACHE_HOME=<short executable path>/xdg
   ```

   The OpenR1 HPT main launcher uses `<repo>/../.rt`, not
   `<repo>/.cache/runtime`, for this reason.

## Rollout Dump Hygiene

For main runs, rollout dumps are useful for later reward and output-quality
analysis. Use `skip.async_rollout.action=dump` for these runs so the model
always generates fresh attempts and the skip layer only writes them to disk.
`cache` is replay-capable and belongs in smoke/debug reuse workflows, not in a
main-run quality dump. The OpenR1 HPT main launcher also keeps the dump directory
unique per run:

```text
<repo>/.cache/rollout_dump/openr1_async_hpt_main_<timestamp>/
```

The async skip step is the Rollouter feed-order index, not trainer
`global_steps`. In HPT trajectory-scheduler mode, all attempts from the same
prompt group share that feed-order step, so the dump path includes the attempt
index:

```text
<feed_step>/attempt_<hpt_rollout_index>/gen_batch.dp
```

The main launcher dumps every async rollout feed step without constructing a
huge Hydra list:

```text
skip.async_rollout.enable=True
skip.async_rollout.action=dump
skip.async_rollout.steps=[]
skip.async_rollout.all_steps=True
```

This is intentional for main-run debugging: the dump should capture post-update
rollouts as well as the initial-policy burst. The disk cost can be large because
every feed step is written; switch `all_steps` back to `False` and provide an
explicit `steps=[...]` list only for bounded smoke/debug runs. For the
staleness/burst interaction with skip/cache, see `AsyncBudget_RL.md`'s
Skip/Cache Caveat.

## Suggested Main-Run Log Checks

During a run, inspect Ray worker logs directly, not only the nohup file. The
nohup file can miss actor-local details.

Find the active Ray session:

```bash
ls -td "${TMPDIR:-/tmp}"/ray/session_* | head -n 1
```

Useful checks:

```bash
SESSION=$(ls -td "${TMPDIR:-/tmp}"/ray/session_* | head -n 1)

rg -n "Loop collection completed|learner_rows|required_multiple|global_steps|_fit_update_weights|Traceback|ERROR|Exception" \
  "$SESSION/logs"/*.out "$SESSION/logs"/*.err

rg -n "MonitorLoop|staleness_samples|pending_queue_size|mq_queue_size|active_tasks_size|hpt/" \
  "$SESSION/logs"/*.out "$SESSION/logs"/*.err
```

Healthy HPT async smoke should show:

```text
Loop collection completed ... learner_rows=<multiple of required_multiple>
global_steps increasing
_fit_update_weights after the configured sync interval
no Traceback / ValueError / AssertionError
```

For interpreting a stalled or deadlocked run from these logs, see
`Codemap_RL.md`'s "Where Did It Break?" index.
