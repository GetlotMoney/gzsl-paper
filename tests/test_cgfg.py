from __future__ import annotations
from pathlib import Path
import torch
from model.candidates.v2.modules.cgfg import ConditionalGaussianFeatureGenerator,CosineLinearClassifier
from model.candidates.v2.trainers.train_cgfg import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_cgfg_starts_at_semantic_and_residual_is_bounded():
    g=torch.Generator().manual_seed(191); semantic=torch.randn(200,768,generator=g); model=ConditionalGaussianFeatureGenerator(semantic); assert torch.allclose(model.means(),torch.nn.functional.normalize(semantic,dim=-1),atol=1e-7); model.network[-1].bias.data.fill_(100.0); assert float(model.residual().norm(dim=-1).max().detach())<=0.200001
def test_cgfg_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_076_cgfg_seed7.yaml"); assert config["idea_id"]=="IDEA-024"; source=(ROOT/"model/candidates/v2/trainers/train_cgfg.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在CGFG训练结束后加载。")
