from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.daln import DensityAwareLogitNormalizer,prototype_density_features
from model.innovations.train_daln import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_daln_starts_off_and_density_is_finite():
    g=torch.Generator().manual_seed(171); prototypes=torch.randn(200,768,generator=g); features=prototype_density_features(prototypes); assert features.shape==(200,4) and torch.isfinite(features).all(); seen=torch.arange(150); model=DensityAwareLogitNormalizer(prototypes,features,seen,torch.tensor(10.0)); assert torch.equal(model.class_confidence(),torch.ones(200)); model.class_confidence().sum().backward(); assert any(p.grad is not None for p in model.gate.parameters())
def test_daln_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_072_daln_seed7.yaml"); assert config["idea_id"]=="IDEA-021"; source=(ROOT/"model/innovations/train_daln.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在DALN训练结束后加载。")
def test_daln_episode_config():
    config,_=load_config(ROOT/"config/tries/v2_try_073_daln_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-073"; assert config["training_objective"]=="pseudo_unseen_episode"; assert config["fold_checkpoint_dir"]
