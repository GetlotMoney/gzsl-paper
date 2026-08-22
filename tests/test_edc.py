from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.edc import EpisodicDomainCompetition,competition_features
from model.innovations.train_edc import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_edc_starts_off_and_features_are_finite():
    g=torch.Generator().manual_seed(161); logits=torch.randn(8,10,generator=g); seen=torch.tensor([True]*6+[False]*4); unseen=~seen; model=EpisodicDomainCompetition(); assert torch.equal(model(logits,seen,unseen),logits); features=competition_features(logits,seen,unseen); assert features.shape==(8,7) and torch.isfinite(features).all(); model(logits,seen,unseen).sum().backward(); assert any(p.grad is not None for p in model.parameters())
def test_edc_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_070_edc_seed7.yaml"); assert config["idea_id"]=="IDEA-020"; source=(ROOT/"model/innovations/train_edc.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在EDC训练结束后加载。")
