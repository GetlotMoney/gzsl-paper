from __future__ import annotations
from pathlib import Path
import torch
from model.candidates.v2.modules.fvra import FeatureVisualResidualAdapter
from model.candidates.v2.trainers.train_mfra import first_order_feature_update,load_config
ROOT=Path(__file__).resolve().parents[3]
def test_mfra_first_order_update_keeps_outer_path():
    g=torch.Generator().manual_seed(181); model=FeatureVisualResidualAdapter(max_residual_norm=0.1); x=torch.randn(8,768,generator=g); loss=model(x).square().mean(); fast=first_order_feature_update(model,loss,0.01); assert fast; assert all(v.requires_grad for v in fast.values())
def test_mfra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_075_mfra_seed7.yaml"); assert config["idea_id"]=="IDEA-023"; source=(ROOT/"model/candidates/v2/trainers/train_mfra.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在MFRA训练结束后加载。")
