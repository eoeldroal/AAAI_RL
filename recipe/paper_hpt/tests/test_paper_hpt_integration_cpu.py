# Copyright 2026
#
# INTEGRATION tests against real verl code (CPU): the custom-loss FQN hook, the
# real GRPO advantage on a routed batch, and the full
# build -> route -> advantage -> dual-loss-core pipeline.

import numpy as np
import torch

from recipe.paper_hpt import paper_hpt_routing as routing
from recipe.paper_hpt.paper_hpt_loss import paper_hpt_dual_loss, paper_hpt_dual_loss_core
from recipe.paper_hpt.paper_hpt_trainer import PaperHptConfig, build_sft_rows_by_uid
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.import_utils import load_class_from_fqn


# --------------------------------------------------------------------------- #
# custom-loss hook resolves (engine_workers uses load_class_from_fqn on this FQN)
# --------------------------------------------------------------------------- #
def test_custom_loss_fqn_resolves_to_dual_loss():
    fn = load_class_from_fqn("recipe.paper_hpt.paper_hpt_loss.paper_hpt_dual_loss", "custom actor loss")
    assert fn is paper_hpt_dual_loss


# --------------------------------------------------------------------------- #
# real GRPO advantage on routed rows
# --------------------------------------------------------------------------- #
def test_grpo_advantage_rl_group_centered_drgrpo():
    tlr = torch.zeros(2, 3)
    tlr[0, 2] = 1.0  # p0 row0 correct, row1 wrong
    adv, _ = compute_grpo_outcome_advantage(
        tlr, torch.ones(2, 3), index=np.array(["p0", "p0"], dtype=object), norm_adv_by_std_in_grpo=False
    )
    assert torch.allclose(adv[0], torch.full((3,), 0.5), atol=1e-6)
    assert torch.allclose(adv[1], torch.full((3,), -0.5), atol=1e-6)


def test_grpo_advantage_sft_row_zero_reward_is_zero():
    # SFT rows carry zero reward -> singleton group -> advantage 0 (unused by dual-loss).
    adv, _ = compute_grpo_outcome_advantage(
        torch.zeros(1, 3), torch.ones(1, 3), index=np.array(["sft_p1"], dtype=object),
        norm_adv_by_std_in_grpo=False,
    )
    assert torch.allclose(adv[0], torch.zeros(3), atol=1e-8)


# --------------------------------------------------------------------------- #
# full pipeline: build SFT rows -> route -> real advantage -> dual-loss core
# --------------------------------------------------------------------------- #
def _generated_batch_with_tau():
    n, seqlen, resp = 4, 6, 4
    tgt_resp = torch.zeros(n, resp)
    tgt_resp[:, :3] = 1.0
    tls = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    return DataProto.from_single_dict(
        {
            "input_ids": torch.arange(n * seqlen).reshape(n, seqlen),
            "attention_mask": torch.ones(n, seqlen, dtype=torch.long),
            "position_ids": torch.arange(seqlen).unsqueeze(0).repeat(n, 1),
            "response_mask": torch.ones(n, resp),
            "old_log_probs": torch.zeros(n, resp),
            "token_level_scores": tls,
            "rm_scores": tls.clone(),
            "tgt_input_ids": torch.arange(n * seqlen).reshape(n, seqlen),
            "tgt_attention_mask": torch.ones(n, seqlen, dtype=torch.long),
            "tgt_position_ids": torch.arange(seqlen).unsqueeze(0).repeat(n, 1),
            "tgt_response_mask": tgt_resp,
            "uid": np.array(["p0", "p0", "p1", "p1"], dtype=object),
        }
    )


def test_full_pipeline_build_route_advantage_dualloss():
    from omegaconf import OmegaConf

    batch = _generated_batch_with_tau()
    cfg = PaperHptConfig(
        OmegaConf.create({"paper_hpt": {"enable": True, "gamma": 0.0, "beta": 0.3, "success_value": 1.0}}),
        OmegaConf.create({}),
    )
    sft_rows = build_sft_rows_by_uid(batch, cfg)
    routed = routing.route_generated_batch(batch, {"p1": sft_rows["p1"]}, gamma=0.0)
    assert routed.batch["hpt_is_sft"].tolist() == [False, False, True]
    assert routed.non_tensor_batch["uid"].tolist() == ["p0", "p0", "p1"]

    # real advantage on the routed batch (RL rows centered; SFT row zero-reward singleton)
    routed.batch["token_level_rewards"] = routed.batch["token_level_scores"]
    adv, _ = compute_grpo_outcome_advantage(
        routed.batch["token_level_rewards"], routed.batch["response_mask"],
        index=routed.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=False,
    )
    assert torch.allclose(adv[0][:3], torch.full((3,), 0.5), atol=1e-6)   # p0 RL centered
    assert torch.allclose(adv[1][:3], torch.full((3,), -0.5), atol=1e-6)
    assert torch.allclose(adv[2], torch.zeros_like(adv[2]), atol=1e-8)    # SFT adv unused/zero

    # feed the routed batch to the dual-loss core (as the actor would)
    log_prob = torch.zeros(3, 4, requires_grad=True)
    loss, raw = paper_hpt_dual_loss_core(
        log_prob=log_prob,
        entropy=torch.rand(3, 4),
        response_mask=routed.batch["response_mask"],
        old_log_prob=torch.zeros(3, 4),
        advantages=adv,
        hpt_is_sft=routed.batch["hpt_is_sft"],
        beta=cfg.beta, loss_scale_factor=8, entropy_coeff=0.001,
    )
    loss.backward()
    assert torch.isfinite(loss)
    # RL rows (0,1) get advantage-driven gradient; SFT row (2) gets NLL gradient (nonzero)
    assert log_prob.grad[2].abs().sum().item() > 0.0
    assert raw["hpt/rl_response_token_count"].item() == 8.0  # 2 RL rows * 4 tokens
    assert raw["hpt/sft_response_token_count"].item() == 3.0  # 1 SFT row * 3 demo tokens
