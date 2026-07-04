# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.trainer.ppo.metric_utils import compute_timing_metrics


def _timing_batch() -> DataProto:
    prompts = torch.tensor([[1, 2]], dtype=torch.long)
    responses = torch.tensor([[3, 4]], dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": torch.ones((1, 2), dtype=torch.long),
        }
    )


def test_fully_async_hpt_metrics_aggregate_window_counts_and_ratios():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics(
        {
            "hpt/num_sft_rows": 8,
            "hpt/num_sft": 2,
            "hpt/num_rl_groups": 1,
            "hpt/missing_tau_count": 1,
            "hpt/p_success_zero_count": 1,
            "hpt/offline_data_ratio": 2 / 3,
            "hpt/p_success_zero_ratio": 1 / 3,
            "actor/hpt/sft_response_token_count": 100,
            "fully_async/hpt_collected_queue_samples": 3,
            "fully_async/hpt_required_training_multiple": 16,
            "fully_async/count/current_param_version": 7,
        },
        sample_count=3,
    )
    aggregator.add_step_metrics(
        {
            "hpt/num_sft_rows": 4,
            "hpt/num_sft": 1,
            "hpt/num_rl_groups": 3,
            "hpt/missing_tau_count": 2,
            "hpt/p_success_zero_count": 3,
            "hpt/offline_data_ratio": 1 / 4,
            "hpt/p_success_zero_ratio": 3 / 4,
            "actor/hpt/sft_response_token_count": 80,
            "fully_async/hpt_collected_queue_samples": 4,
            "fully_async/hpt_required_training_multiple": 32,
            "fully_async/count/current_param_version": 8,
        },
        sample_count=4,
    )

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["hpt/num_sft_rows"] == 12
    assert metrics["hpt/num_sft"] == 3
    assert metrics["hpt/num_rl_groups"] == 4
    assert metrics["hpt/missing_tau_count"] == 3
    assert metrics["hpt/p_success_zero_count"] == 4
    assert metrics["hpt/offline_data_ratio"] == pytest.approx(3 / 7)
    assert metrics["hpt/p_success_zero_ratio"] == pytest.approx(4 / 7)
    assert metrics["actor/hpt/sft_response_token_count"] == 180
    assert metrics["fully_async/hpt_collected_queue_samples"] == 7
    assert metrics["fully_async/hpt_required_training_multiple"] == 32
    assert metrics["fully_async/count/current_param_version"] == 8


def test_timing_metrics_do_not_double_prefix_prefixed_timing_keys():
    metrics = compute_timing_metrics(
        batch=_timing_batch(),
        timing_raw={
            "timing_s/param_sync": 2.0,
            "step": 5.0,
        },
    )

    assert metrics["timing_s/param_sync"] == 2.0
    assert metrics["timing_s/step"] == 5.0
    assert "timing_s/timing_s/param_sync" not in metrics


def test_fully_async_mean_metrics_use_metric_local_sample_weights():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics({"training/global_step": 1}, sample_count=100)
    aggregator.add_step_metrics({"critic/score/mean": 10.0}, sample_count=2)
    aggregator.add_step_metrics({"critic/score/mean": 30.0}, sample_count=6)

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["critic/score/mean"] == pytest.approx((10.0 * 2 + 30.0 * 6) / 8)
    assert metrics["training/global_step"] == 1


def test_fully_async_loss_kl_fraction_metrics_are_weighted_and_state_metrics_use_last():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics(
        {
            "actor/pg_loss": 1.0,
            "actor/ppo_kl": 0.5,
            "rollout_is_ratio_fraction_high": 0.25,
            "training/epoch": 1,
            "actor/lr": 1e-6,
        },
        sample_count=2,
    )
    aggregator.add_step_metrics(
        {
            "actor/pg_loss": 3.0,
            "actor/ppo_kl": 1.5,
            "rollout_is_ratio_fraction_high": 0.75,
            "training/epoch": 2,
            "actor/lr": 2e-6,
        },
        sample_count=6,
    )

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["actor/pg_loss"] == pytest.approx((1.0 * 2 + 3.0 * 6) / 8)
    assert metrics["actor/ppo_kl"] == pytest.approx((0.5 * 2 + 1.5 * 6) / 8)
    assert metrics["rollout_is_ratio_fraction_high"] == pytest.approx((0.25 * 2 + 0.75 * 6) / 8)
    assert metrics["training/epoch"] == 2
    assert metrics["actor/lr"] == pytest.approx(2e-6)


def test_fully_async_explicit_weighted_metrics_without_mean_keywords():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics(
        {
            "actor/entropy": 2.0,
            "actor/hpt/sft_nll": 10.0,
            "hpt/sft_pseudo_reward/mean": 0.3,
            "rollout_is_eff_sample_size": 0.5,
            "_metric_weight/actor/entropy": 2,
            "_metric_weight/actor/hpt/sft_nll": 1,
            "_metric_weight/hpt/sft_pseudo_reward/mean": 1,
            "_metric_weight/rollout_is_eff_sample_size": 4,
        },
        sample_count=100,
    )
    aggregator.add_step_metrics(
        {
            "actor/entropy": 6.0,
            "actor/hpt/sft_nll": 30.0,
            "hpt/sft_pseudo_reward/mean": 0.9,
            "rollout_is_eff_sample_size": 0.25,
            "_metric_weight/actor/entropy": 6,
            "_metric_weight/actor/hpt/sft_nll": 3,
            "_metric_weight/hpt/sft_pseudo_reward/mean": 3,
            "_metric_weight/rollout_is_eff_sample_size": 4,
        },
        sample_count=100,
    )

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["actor/entropy"] == pytest.approx((2.0 * 2 + 6.0 * 6) / 8)
    assert metrics["actor/hpt/sft_nll"] == pytest.approx((10.0 * 1 + 30.0 * 3) / 4)
    assert metrics["hpt/sft_pseudo_reward/mean"] == pytest.approx((0.3 * 1 + 0.9 * 3) / 4)
    assert metrics["rollout_is_eff_sample_size"] == pytest.approx((0.5 * 4 + 0.25 * 4) / 8)


def test_fully_async_min_max_rules_take_priority_over_weighted_keywords():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics(
        {
            "log_ppl_diff_max": 2.0,
            "rollout_rs_exact_min": 0.75,
            "_metric_weight/log_ppl_diff_max": 100,
            "_metric_weight/rollout_rs_exact_min": 100,
        },
        sample_count=100,
    )
    aggregator.add_step_metrics(
        {
            "log_ppl_diff_max": 1.0,
            "rollout_rs_exact_min": 0.25,
            "_metric_weight/log_ppl_diff_max": 1,
            "_metric_weight/rollout_rs_exact_min": 1,
        },
        sample_count=1,
    )

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["log_ppl_diff_max"] == 2.0
    assert metrics["rollout_rs_exact_min"] == 0.25


def test_fully_async_mean_metrics_can_use_metric_specific_hidden_weights():
    aggregator = MetricsAggregator(total_gpus=4)
    aggregator.add_step_metrics(
        {
            "critic/advantages/mean": 10.0,
            "_metric_weight/critic/advantages/mean": 2,
        },
        sample_count=100,
    )
    aggregator.add_step_metrics(
        {
            "critic/advantages/mean": 30.0,
            "_metric_weight/critic/advantages/mean": 6,
        },
        sample_count=100,
    )

    metrics = aggregator.get_aggregated_metrics()

    assert metrics["critic/advantages/mean"] == pytest.approx((10.0 * 2 + 30.0 * 6) / 8)
    assert "_metric_weight/critic/advantages/mean" not in metrics


def test_fully_async_postprocess_uses_actual_learner_row_count_for_metric_weighting():
    class CapturingAggregator:
        def __init__(self):
            self.sample_count = None

        def add_step_metrics(self, *, metrics, sample_count, timestamp):
            self.sample_count = sample_count

    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.global_steps = 3
    trainer.local_trigger_step = 2
    trainer.metrics = {"critic/score/mean": 1.0}
    trainer.metrics_aggregator = CapturingAggregator()
    batch = DataProto.from_dict(tensors={"responses": torch.ones((5, 2), dtype=torch.long)})

    trainer._fit_postprocess_step(batch)

    assert trainer.metrics_aggregator.sample_count == 5
    assert trainer.global_steps == 4


def test_fully_async_metric_weight_sidecars_match_sequence_and_token_denominators():
    trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = object.__new__(trainer_cls)
    trainer.use_critic = True
    trainer.metrics = {
        "actor/pg_loss": 0.0,
        "actor/entropy": 0.0,
        "actor/entropy_loss": 0.0,
        "actor/hpt/sft_nll": 0.0,
        "hpt/sft_pseudo_reward/mean": 0.0,
        "critic/score/mean": 0.0,
        "critic/advantages/mean": 0.0,
        "critic/values/mean": 0.0,
        "rollout_is_seq_mean": 0.0,
        "rollout_rs_exact_mean": 0.0,
        "rollout_rs_exact_seq_fraction_high": 0.0,
        "response/aborted_ratio": 0.0,
        "response_length_non_aborted/mean": 0.0,
    }
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((3, 2), dtype=torch.long),
            "responses": torch.ones((3, 3), dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.long,
            ),
            "hpt_is_sft": torch.tensor([True, False, True], dtype=torch.bool),
        }
    )

    trainer._collect_metric_aggregation_weights(batch)

    assert trainer.metrics["_metric_weight/actor/pg_loss"] == 4
    assert trainer.metrics["_metric_weight/actor/entropy"] == 1
    assert trainer.metrics["_metric_weight/actor/entropy_loss"] == 1
    assert trainer.metrics["_metric_weight/actor/hpt/sft_nll"] == 3
    assert trainer.metrics["_metric_weight/hpt/sft_pseudo_reward/mean"] == 2
    assert trainer.metrics["_metric_weight/critic/score/mean"] == 2
    assert trainer.metrics["_metric_weight/response_length_non_aborted/mean"] == 2
    assert trainer.metrics["_metric_weight/critic/advantages/mean"] == 4
    assert trainer.metrics["_metric_weight/critic/values/mean"] == 4
    assert trainer.metrics["_metric_weight/rollout_is_seq_mean"] == 3
    assert trainer.metrics["_metric_weight/rollout_rs_exact_mean"] == 4
    assert trainer.metrics["_metric_weight/rollout_rs_exact_seq_fraction_high"] == 3
    assert trainer.metrics["_metric_weight/response/aborted_ratio"] == 3
