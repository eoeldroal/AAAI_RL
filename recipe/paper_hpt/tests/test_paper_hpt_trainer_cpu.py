# Copyright 2026
#
# Trainer-logic tests (CPU, no ray/GPU): PaperHptConfig, build_sft_rows_by_uid,
# and PaperHptTrainer._paper_hpt_route (via __new__ so RayPPOTrainer.__init__ is
# skipped). With the explicit dual-loss, SFT rows carry NO synthetic reward.

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.paper_hpt.paper_hpt_trainer import (
    PAPER_HPT_BETA_KEY,
    PaperHptConfig,
    PaperHptTrainer,
    build_sft_rows_by_uid,
)
from verl.protocol import DataProto


# --------------------------------------------------------------------------- #
# PaperHptConfig
# --------------------------------------------------------------------------- #
def test_config_parses_values():
    algo = OmegaConf.create({"paper_hpt": {"enable": True, "gamma": 0.25, "beta": 0.3, "success_value": 1.0}})
    cfg = PaperHptConfig(algo, OmegaConf.create({}))
    assert (cfg.enable, cfg.gamma, cfg.beta, cfg.success_value) == (True, 0.25, 0.3, 1.0)


def test_config_defaults_when_absent():
    cfg = PaperHptConfig(OmegaConf.create({}), OmegaConf.create({}))
    assert cfg.enable is False
    assert cfg.gamma == 0.0
    assert cfg.beta == 0.3
    assert cfg.success_value == 1.0


# --------------------------------------------------------------------------- #
# build_sft_rows_by_uid
# --------------------------------------------------------------------------- #
def _tau_batch(uids):
    n, seqlen, resp = len(uids), 5, 3
    tgt_resp_mask = torch.zeros(n, resp)
    tgt_resp_mask[:, :2] = 1.0
    return DataProto.from_single_dict(
        {
            "tgt_input_ids": torch.arange(n * seqlen).reshape(n, seqlen),
            "tgt_attention_mask": torch.ones(n, seqlen, dtype=torch.long),
            "tgt_position_ids": torch.arange(seqlen).unsqueeze(0).repeat(n, 1),
            "tgt_response_mask": tgt_resp_mask,
            "uid": np.array(uids, dtype=object),
        }
    )


def _cfg():
    return PaperHptConfig(OmegaConf.create({"paper_hpt": {"enable": True, "beta": 0.3}}), OmegaConf.create({}))


def test_build_sft_rows_one_per_uid_no_synthetic_reward():
    rows = build_sft_rows_by_uid(_tau_batch(["p0", "p0", "p1", "p1"]), _cfg())
    assert set(rows.keys()) == {"p0", "p1"}
    r = rows["p1"]
    assert len(r) == 1
    assert bool(r.batch["hpt_is_sft"].all())
    # SFT rows carry NO reward (dual-loss uses masked-mean NLL, not advantage)
    assert r.batch["token_level_scores"].abs().sum().item() == 0.0
    assert r.batch["rm_scores"].abs().sum().item() == 0.0
    # demonstration tensors come from the tau fields
    assert r.batch["response_mask"].tolist() == [[1.0, 1.0, 0.0]]


def test_build_sft_rows_missing_tau_raises():
    batch = DataProto.from_single_dict(
        {"input_ids": torch.zeros(1, 3, dtype=torch.long), "uid": np.array(["p0"], dtype=object)}
    )
    with pytest.raises(KeyError):
        build_sft_rows_by_uid(batch, _cfg())


# --------------------------------------------------------------------------- #
# _paper_hpt_route
# --------------------------------------------------------------------------- #
def _trainer(enable):
    t = PaperHptTrainer.__new__(PaperHptTrainer)  # skip RayPPOTrainer.__init__
    t.config = OmegaConf.create(
        {
            "algorithm": {"paper_hpt": {"enable": enable, "gamma": 0.0, "beta": 0.3, "success_value": 1.0}},
            "actor_rollout_ref": {"actor": {}},
        }
    )
    return t


def _routable_batch():
    n, seqlen, resp = 4, 5, 3
    tgt_resp_mask = torch.zeros(n, resp)
    tgt_resp_mask[:, :2] = 1.0
    tls = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return DataProto.from_single_dict(
        {
            # RL-row training schema (matches build_sft_rows output after tau is dropped)
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
            "tgt_response_mask": tgt_resp_mask,
            "uid": np.array(["p0", "p0", "p1", "p1"], dtype=object),
        }
    )


def test_route_disabled_is_noop():
    out = _trainer(enable=False)._paper_hpt_route(_routable_batch())
    assert len(out) == 4
    assert "hpt_is_sft" not in out.batch


def test_route_enabled_reconstructs_and_injects_beta_meta():
    out = _trainer(enable=True)._paper_hpt_route(_routable_batch())
    assert len(out) == 3  # p0 2 RL rows + p1 1 SFT row
    assert out.batch["hpt_is_sft"].tolist() == [False, False, True]
    # beta is injected into meta so the dual-loss can read it
    assert out.meta_info[PAPER_HPT_BETA_KEY] == pytest.approx(0.3)
