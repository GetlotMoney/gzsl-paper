from __future__ import annotations
from pathlib import Path
import torch
from model.candidates.v2.modules.fvra import FeatureVisualResidualAdapter
from model.candidates.v2.trainers.train_fvra import load_config
ROOT=Path(__file__).resolve().parents[3]
def test_fvra_starts_as_normalized_parent_and_trains():
    g=torch.Generator().manual_seed(151); x=torch.randn(8,768,generator=g); model=FeatureVisualResidualAdapter(); assert torch.allclose(model(x),model(x,enabled=False),atol=1e-7); model(x).square().mean().backward(); assert any(p.grad is not None for p in model.network.parameters())
def test_fvra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_068_fvra_seed7.yaml"); assert config["idea_id"]=="IDEA-019"; source=(ROOT/"model/candidates/v2/trainers/train_fvra.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在FVRA训练结束后加载。")
def test_bounded_fvra_and_rescue_config():
    g=torch.Generator().manual_seed(152); x=torch.randn(8,768,generator=g); model=FeatureVisualResidualAdapter(max_residual_norm=0.1); model.network[-1].bias.data.fill_(100.0); normalized=torch.nn.functional.normalize(x,dim=-1); assert float(model.residual_vectors(normalized).norm(dim=-1).max().detach())<=0.100001
    config,_=load_config(ROOT/"config/tries/v2_try_069_fvra_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-069"; assert config["consistency_weight"]==1.0; assert config["max_residual_norm"]==0.1
