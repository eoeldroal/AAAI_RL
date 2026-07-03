import json
from pathlib import Path


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
