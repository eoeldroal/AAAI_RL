import json
import re
import shutil
import shlex
import subprocess
from pathlib import Path

import pytest


def test_openr1_hpt_preprocess_builds_unified_prompt_rows_with_tau_messages():
    from examples.data_preprocess.openr1_hpt import build_hpt_rows

    raw_rows = [
        {
            "data_source": "olympiads",
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "What is 1+1?"},
            ],
            "target": [{"role": "assistant", "content": "The answer is \\boxed{2}."}],
            "ability": "",
            "reward_model": {"style": "rule", "ground_truth": "2"},
            "extra_info": {"split": "default", "index": -1},
        },
        {
            "data_source": "numina_amc_aime",
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "target": [{"role": "assistant", "content": "The answer is \\boxed{4}."}],
            "ability": "",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "default", "index": -1},
        },
    ]

    train_rows = build_hpt_rows(
        raw_rows,
        split="train",
        prompt_uid_prefix="openr1_smoke",
        normalize_data_source="numina_olympiads",
        strip_system_prompt=True,
    )

    assert [row["prompt_uid"] for row in train_rows] == ["openr1_smoke_train_00000000", "openr1_smoke_train_00000001"]
    assert all(row["data_source"] == "numina_olympiads" for row in train_rows)
    assert train_rows[0]["prompt"] == [{"role": "user", "content": "What is 1+1?"}]
    assert train_rows[0]["extra_info"]["source_data_source"] == "olympiads"
    assert train_rows[0]["extra_info"]["prompt_uid"] == "openr1_smoke_train_00000000"

    assert "tau_messages" in train_rows[0]
    assert "tau_messages" in train_rows[1]
    assert json.loads(train_rows[0]["tau_messages"]) == [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "The answer is \\boxed{2}."},
    ]
    assert json.loads(train_rows[1]["tau_messages"]) == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is \\boxed{4}."},
    ]


def test_openr1_hpt_preprocess_can_build_full_main_rows_without_selection_loss():
    from examples.data_preprocess.openr1_hpt import build_hpt_rows, select_openr1_rows_for_hpt

    raw_rows = [
        {
            "data_source": "olympiads",
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": f"What is {idx}+{idx}?"},
            ],
            "target": [{"role": "assistant", "content": f"The answer is \\boxed{{{idx + idx}}}."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(idx + idx)},
            "extra_info": {"split": "default", "index": idx},
        }
        for idx in range(3)
    ]

    selected = select_openr1_rows_for_hpt(
        raw_rows,
        total_rows=-1,
        max_target_chars=-1,
        reward_data_source="numina_olympiads",
        tokenizer_path=None,
        max_prompt_tokens=1024,
        max_response_tokens=8192,
        strip_system_prompt=True,
    )
    train_rows = build_hpt_rows(
        selected,
        split="train",
        prompt_uid_prefix="openr1_hpt_main",
        normalize_data_source="numina_olympiads",
        strip_system_prompt=True,
    )

    assert len(train_rows) == 3
    assert [row["prompt_uid"] for row in train_rows] == [
        "openr1_hpt_main_train_00000000",
        "openr1_hpt_main_train_00000001",
        "openr1_hpt_main_train_00000002",
    ]
    assert all(row["prompt"][0]["role"] == "user" for row in train_rows)
    assert all(json.loads(row["tau_messages"])[-1]["role"] == "assistant" for row in train_rows)


def test_unify_eval_json_preprocess_matches_verl_reward_schema():
    from examples.data_preprocess.openr1_hpt import build_unify_eval_rows

    raw_rows = [
        {"prompt": "Solve x+1=2.", "answer": "1"},
        {"prompt": "Find 2+2.", "answer": "4", "source": "math", "id": "test/algebra/1.json"},
    ]

    rows = build_unify_eval_rows(raw_rows, split="test", data_source="MATH-500")

    assert rows == [
        {
            "data_source": "MATH-500",
            "prompt": [{"role": "user", "content": "Solve x+1=2."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "1"},
            "extra_info": {"split": "test", "index": "MATH-500-0"},
        },
        {
            "data_source": "MATH-500",
            "prompt": [{"role": "user", "content": "Find 2+2."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "test", "index": "MATH-500-test/algebra/1.json", "source": "math"},
        },
    ]


def test_openr1_hpt_preprocess_rejects_rows_outside_token_limits():
    from examples.data_preprocess.openr1_hpt import _within_token_limits

    class PrefixStableTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize
            text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                text += f"<{role}>"
                if role != "assistant":
                    text += content + f"</{role}>"
                else:
                    text += content
            if add_generation_prompt:
                text += "<assistant>"
            return list(range(len(text)))

    row = {
        "prompt": [{"role": "user", "content": "short"}],
        "target": [{"role": "assistant", "content": "answer"}],
    }
    tokenizer = PrefixStableTokenizer()

    assert _within_token_limits(
        row,
        token_counter=tokenizer,
        max_prompt_tokens=64,
        max_response_tokens=32,
        strip_system_prompt=True,
    )
    assert not _within_token_limits(
        row,
        token_counter=tokenizer,
        max_prompt_tokens=4,
        max_response_tokens=32,
        strip_system_prompt=True,
    )
    assert not _within_token_limits(
        row,
        token_counter=tokenizer,
        max_prompt_tokens=64,
        max_response_tokens=4,
        strip_system_prompt=True,
    )


# ---------------------------------------------------------------------------
# Main-run launcher contract
#
# These assertions are deliberately VALUE-FREE. A main-run launcher is a run
# choice, not a code contract: batch sizes, staleness, and require_batches
# change per experiment. The sync<->async mapping lives in docs/Codemap_RL.md
# and docs/AsyncBudget_RL.md, not here.
#
# We assert only that every launcher in main_scripts/ still composes into a
# *valid* async-HPT config and fails closed on a malformed one. Concrete values
# stay in the launcher (the source of truth). This is what stops a value edit
# from silently drifting the test out of agreement with the launcher.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAIN_SCRIPTS_DIR = _REPO_ROOT / "main_scripts"
_FULLY_ASYNC_CONFIG_DIR = _REPO_ROOT / "verl/experimental/fully_async_policy/config"
_FULLY_ASYNC_ENTRYPOINT = "python3 -m verl.experimental.fully_async_policy.fully_async_main"
_SHELL_VAR = re.compile(r"\$\{[^}]+\}|\$\w+")


def _extract_fully_async_overrides(script_text):
    """Return the Hydra overrides from a launcher's fully_async_main invocation.

    Returns None if the launcher does not invoke fully_async_main directly.
    Unresolved shell variables are replaced with a placeholder path because we
    validate config structure, not runtime paths.
    """
    lines = []
    capturing = False
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if line.startswith(_FULLY_ASYNC_ENTRYPOINT):
            capturing = True
        if not capturing:
            continue
        if line == '"$@"':
            break
        if line.endswith("\\"):
            line = line[:-1].strip()
        lines.append(line)
    if not lines:
        return None
    command = _SHELL_VAR.sub("/tmp/placeholder", " ".join(lines))
    argv = shlex.split(command)
    assert argv[:3] == _FULLY_ASYNC_ENTRYPOINT.split()
    return argv[3:]


def _main_launcher_params():
    scripts = sorted(_MAIN_SCRIPTS_DIR.glob("*.sh")) if _MAIN_SCRIPTS_DIR.is_dir() else []
    if not scripts:
        return [pytest.param(None, marks=pytest.mark.skip(reason="no main_scripts/*.sh launchers found"))]
    return [pytest.param(path, id=path.name) for path in scripts]


def _write_fake_main_launcher_env(tmp_path: Path) -> Path:
    """Create the minimum fake conda env needed to execute the launcher to python argv capture."""
    # The cluster /tmp can be mounted noexec. Put the fake executable under the
    # repo-local cache so this contract test exercises the launcher's real exec path.
    fake_home = _REPO_ROOT / ".cache" / "pytest_launcher_env" / tmp_path.name / "home"
    if fake_home.exists():
        shutil.rmtree(fake_home)
    fake_env = fake_home / "miniconda3" / "envs" / "RL"
    fake_bin = fake_env / "bin"
    fake_hook_dir = fake_env / "etc" / "conda" / "activate.d"
    fake_bin.mkdir(parents=True)
    fake_hook_dir.mkdir(parents=True)
    (fake_hook_dir / "verl_cuda_stack.sh").write_text("# fake activation hook for launcher contract tests\n")
    (fake_bin / "python3").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${VERL_TEST_CAPTURE_ARGV}"
""",
        encoding="utf-8",
    )
    (fake_bin / "python3").chmod(0o755)
    return fake_home


@pytest.mark.parametrize("script_path", _main_launcher_params())
def test_main_launcher_composes_to_valid_async_hpt_config(script_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.skip.config import SkipManagerConfig

    overrides = _extract_fully_async_overrides(script_path.read_text())
    if overrides is None:
        pytest.skip(f"{script_path.name} does not invoke fully_async_main directly")

    with initialize_config_dir(config_dir=str(_FULLY_ASYNC_CONFIG_DIR.resolve()), version_base=None):
        config = compose(config_name="fully_async_ppo_trainer", overrides=overrides)

    OmegaConf.resolve(config)

    # Fail closed on a launcher that violates the async-HPT contract.
    validate_async_hpt_config(config)
    config = migrate_legacy_reward_impl(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    # Structural shape of the HPT-on-GRPO objective (derived, not experiment values).
    assert not need_critic(config)
    assert not need_reference_policy(config)

    # Dataclass conversion catches keys that Hydra accepts syntactically but
    # that the actual worker/server configuration objects cannot consume.
    omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine)
    skip_config = omega_conf_to_dataclass(config.skip, SkipManagerConfig)
    assert skip_config.async_rollout.enable is True
    assert skip_config.async_rollout.action == "dump"
    assert skip_config.async_rollout.steps == []
    assert skip_config.async_rollout.all_steps is True


def test_openr1_main_launcher_executes_to_fully_async_entrypoint_with_dump_and_wandb(tmp_path):
    script_path = _MAIN_SCRIPTS_DIR / "run_fully_async_policy_openr1_hpt_main.sh"
    capture_path = tmp_path / "argv.txt"
    fake_home = _write_fake_main_launcher_env(tmp_path)
    env = {
        **dict(),
        "HOME": str(fake_home),
        "PATH": "/usr/bin:/bin",
        "VERL_TEST_CAPTURE_ARGV": str(capture_path),
    }

    result = subprocess.run(
        ["bash", str(script_path), "trainer.total_training_steps=1"],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = capture_path.read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-m", "verl.experimental.fully_async_policy.fully_async_main"]
    assert "trainer.logger=['console','wandb']" in argv
    assert "skip.async_rollout.action=dump" in argv
    assert "skip.async_rollout.steps=[]" in argv
    assert "skip.async_rollout.all_steps=True" in argv
