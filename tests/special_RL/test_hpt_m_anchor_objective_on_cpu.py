import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.fully_async_policy.hpt_config import validate_async_hpt_config
from verl.experimental.fully_async_policy.hpt_training import should_use_hpt_rollout_logprob_anchor
from verl.trainer.ppo.core_algos import compute_policy_loss_cispo
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


def _make_hpt_config(
    *,
    rl_old_logprob_source: str = "entry",
    rollout_is: str | None = "token",
    rollout_rs: str | None = None,
    loss_mode: str = "cispo",
    cispo_clip_mode: str = "upper",
    cispo_epsilon_high: float = 5.0,
):
    return OmegaConf.create(
        {
            "async_hpt": {
                "enabled": True,
                "tau_dataset_path": "/tmp/tau.parquet",
                "tau_messages_key": "tau_messages",
                "gamma": 0.0,
                "alpha": 1.0,
                "beta": 1.0,
                "sft_beta_mode": "constant",
                "loss_aggregation": "branch_blind",
                "sft_entropy_enabled": False,
                "sft_kl_enabled": False,
                "fail_on_missing_tau": True,
                "rl_old_logprob_source": rl_old_logprob_source,
                "entry_proximal": "recent",
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": False,
                "rollout_correction": {
                    "rollout_is": rollout_is,
                    "rollout_rs": rollout_rs,
                    "rollout_is_threshold": 2.0,
                    "bypass_mode": False,
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "calculate_log_probs": True,
                    "n": 8,
                },
                "actor": {
                    "loss_agg_mode": "seq-mean-token-sum-norm",
                    "loss_scale_factor": 8192,
                    "policy_loss": {
                        "loss_mode": loss_mode,
                        "cispo_clip_mode": cispo_clip_mode,
                        "cispo_epsilon_high": cispo_epsilon_high,
                    },
                },
            },
        }
    )


def _make_actor_config(*, cispo_clip_mode: str = "upper", cispo_epsilon_high: float = 5.0) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_mini_batch_size=1,
        ppo_micro_batch_size=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=10.0,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        global_batch_info={"dp_size": 1},
        policy_loss={
            "loss_mode": "cispo",
            "cispo_clip_mode": cispo_clip_mode,
            "cispo_epsilon_high": cispo_epsilon_high,
        },
    )


def test_hpt_entry_recent_config_requires_token_tis_and_upper_cispo():
    validated = validate_async_hpt_config(_make_hpt_config())
    assert validated.rl_old_logprob_source == "entry"
    assert validated.entry_proximal == "recent"

    with pytest.raises(ValueError, match="rollout_is"):
        validate_async_hpt_config(_make_hpt_config(rollout_is=None))

    with pytest.raises(ValueError, match="rollout_rs"):
        validate_async_hpt_config(_make_hpt_config(rollout_rs="token"))

    with pytest.raises(ValueError, match="cispo_clip_mode"):
        validate_async_hpt_config(_make_hpt_config(cispo_clip_mode="two_sided"))

    with pytest.raises(ValueError, match="cispo_epsilon_high"):
        validate_async_hpt_config(_make_hpt_config(cispo_epsilon_high=0.5))


def test_hpt_entry_source_recomputes_old_log_probs_instead_of_using_rollout_anchor():
    batch = DataProto.from_dict(
        tensors={
            "hpt_is_sft": torch.tensor([False]),
            "response_mask": torch.ones(1, 2, dtype=torch.bool),
            "rollout_log_probs": torch.zeros(1, 2),
        }
    )

    assert should_use_hpt_rollout_logprob_anchor(_make_hpt_config(rl_old_logprob_source="rollout"), batch)
    assert not should_use_hpt_rollout_logprob_anchor(_make_hpt_config(rl_old_logprob_source="entry"), batch)


class _RecordingActorWorker:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def save_model_to_cpu(self, version: int):
        self.calls.append(("save", version))

    def restore_model_from_cpu(self, version: int):
        self.calls.append(("restore", version))

    def clear_cpu_model(self, version: int):
        self.calls.append(("clear", version))


def test_fully_async_entry_recent_old_logprob_bypasses_mis_weight_restore(monkeypatch):
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.config = _make_hpt_config(rl_old_logprob_source="entry")
    trainer.local_trigger_step = 3
    trainer.actor_rollout_wg = _RecordingActorWorker()

    sentinel_batch = object()

    def fake_compute_old_log_prob(self, batch):
        assert batch is sentinel_batch
        return "recent-old-logprob", 0.5

    monkeypatch.setattr(SeparateRayPPOTrainer, "_compute_old_log_prob", fake_compute_old_log_prob)

    old_log_prob, mfu = trainer._compute_old_log_prob(sentinel_batch)

    assert old_log_prob == "recent-old-logprob"
    assert mfu == 0.5
    assert trainer.actor_rollout_wg.calls == []


def test_fully_async_rollout_source_keeps_existing_mis_weight_restore(monkeypatch):
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.config = _make_hpt_config(rl_old_logprob_source="rollout")
    trainer.local_trigger_step = 3
    trainer.actor_rollout_wg = _RecordingActorWorker()

    def fake_compute_old_log_prob(self, batch):
        return "mis-old-logprob", 0.25

    monkeypatch.setattr(SeparateRayPPOTrainer, "_compute_old_log_prob", fake_compute_old_log_prob)

    old_log_prob, mfu = trainer._compute_old_log_prob(object())

    assert old_log_prob == "mis-old-logprob"
    assert mfu == 0.25
    assert trainer.actor_rollout_wg.calls == [("save", 3), ("restore", 1), ("restore", 3), ("clear", 3)]


def test_cispo_upper_only_uses_absolute_cap_without_lower_floor():
    config = _make_actor_config(cispo_clip_mode="upper", cispo_epsilon_high=5.0)
    ratio = torch.tensor([[0.1, 2.0, 10.0]])
    old_log_prob = torch.zeros_like(ratio)
    log_prob = torch.log(ratio).detach().clone().requires_grad_(True)
    advantages = torch.ones_like(ratio)
    response_mask = torch.ones_like(ratio, dtype=torch.bool)

    pg_loss, metrics = compute_policy_loss_cispo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    expected_coeff = torch.tensor([[0.1, 2.0, 5.0]])
    expected_loss = (-(expected_coeff * advantages * log_prob)).mean()
    assert pg_loss.item() == pytest.approx(expected_loss.item())
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0 / 3.0)

    pg_loss.backward()
    assert torch.allclose(log_prob.grad, -expected_coeff / 3.0)


def _make_hpt_cispo_batch(*, sft_old_value: float) -> TensorDict:
    response_mask = torch.ones(2, 2, dtype=torch.bool)
    attention_mask = torch.ones(2, 4, dtype=torch.bool)
    input_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    position_ids = torch.arange(4, dtype=torch.long).repeat(2, 1)
    old_log_probs = torch.tensor([[sft_old_value, sft_old_value], [0.0, 0.0]])
    rollout_is_weights = torch.tensor([[99.0, 99.0], [2.0, 2.0]])

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, -2:],
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": torch.ones(2, 2),
            "loss_mask": response_mask.clone(),
            "loss_scale": torch.ones(2, 2),
            "rollout_is_weights": rollout_is_weights,
            "hpt_is_sft": torch.tensor([True, False]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        batch,
        dp_size=1,
        batch_num_tokens=int(response_mask.sum().item()),
        global_batch_size=2,
    )
    return batch


def test_hpt_cispo_preserves_sft_self_detach_and_ignores_sft_rollout_is_weight():
    config = _make_actor_config(cispo_clip_mode="upper", cispo_epsilon_high=5.0)
    model_output = {"log_probs": torch.full((8,), -0.25)}

    low_old_loss, _ = ppo_loss(
        config=config,
        model_output=model_output,
        data=_make_hpt_cispo_batch(sft_old_value=-20.0),
    )
    high_old_loss, _ = ppo_loss(
        config=config,
        model_output=model_output,
        data=_make_hpt_cispo_batch(sft_old_value=20.0),
    )

    assert torch.allclose(low_old_loss, high_old_loss)
