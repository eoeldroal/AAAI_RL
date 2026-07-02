import json
from pathlib import Path


def test_openr1_hpt_preprocess_builds_train_and_partial_tau_rows():
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

    train_rows, tau_rows = build_hpt_rows(
        raw_rows,
        split="train",
        prompt_uid_prefix="openr1_smoke",
        normalize_data_source="numina_olympiads",
        strip_system_prompt=True,
        tau_keep_every=2,
    )

    assert [row["prompt_uid"] for row in train_rows] == ["openr1_smoke_train_00000000", "openr1_smoke_train_00000001"]
    assert all(row["data_source"] == "numina_olympiads" for row in train_rows)
    assert train_rows[0]["prompt"] == [{"role": "user", "content": "What is 1+1?"}]
    assert train_rows[0]["extra_info"]["source_data_source"] == "olympiads"
    assert train_rows[0]["extra_info"]["prompt_uid"] == "openr1_smoke_train_00000000"

    assert len(tau_rows) == 1
    assert tau_rows[0]["prompt_uid"] == "openr1_smoke_train_00000000"
    tau_messages = json.loads(tau_rows[0]["tau_messages"])
    assert tau_messages == [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "The answer is \\boxed{2}."},
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


def test_sglang_smoke_script_uses_openr1_hpt_dataset_and_enables_async_hpt():
    script = Path("tests/special_e2e/run_fully_async_policy_sglang_smoke.sh").read_text()

    assert "examples/data_preprocess/openr1_hpt.py" in script
    assert "OPENR1_HPT_DATA_DIR" in script
    assert "OPENR1_HPT_TAU_FILE" in script
    assert "async_hpt.enabled=True" in script
    assert 'async_hpt.tau_dataset_path="${OPENR1_HPT_TAU_FILE}"' in script
    assert "async_hpt.trajectory_scheduler.enabled=True" in script
    assert "algorithm.norm_adv_by_std_in_grpo=False" in script
    assert "async_hpt.gamma=1.0" in script
    assert "--normalize_data_source numina_olympiads" in script
    assert "--tokenizer_path \"${MODEL_PATH}\"" in script
    assert "--max_prompt_tokens 1024" in script
    assert "--max_response_tokens 2048" in script
    assert "export N_RESP_PER_PROMPT=4" in script
    assert "export TRAIN_PROMPT_MINI_BSZ=4" in script
    assert "export PARTIAL_ROLLOUT=True" in script
    assert "export STALENESS_THRESHOLD=1.0" in script
    assert "export TRAINER_TOTAL_EPOCHS=30" in script
    assert "export TOTAL_ROLLOUT_STEPS=480" in script
    assert "async_training.partial_rollout=True" in script
    assert "async_training.max_inflight_prompt_groups=16" in script
    assert "async_training.max_completed_prompt_groups=32" in script
