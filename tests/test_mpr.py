from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.mpr import MultiRolePrototypeClassifier
from model.innovations.train_mpr import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_mpr_off_is_parent_and_strength_trainable():
    g=torch.Generator().manual_seed(111); parent=torch.randn(200,768,generator=g); roles=torch.randn(200,3,768,generator=g); model=MultiRolePrototypeClassifier(parent,roles,torch.tensor(10.0)); images=torch.randn(6,768,generator=g); expected=torch.nn.functional.normalize(images,dim=-1)@torch.nn.functional.normalize(parent,dim=-1).T*10.0; assert torch.allclose(model.logits(images,enabled=False),expected,atol=1e-5); torch.nn.functional.cross_entropy(model.logits(images),torch.arange(6)).backward(); assert model.raw_strength.grad is not None
def test_mpr_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_046_mpr_seed7.yaml"); assert config["idea_id"]=="IDEA-014"; source=(ROOT/"model/innovations/train_mpr.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在MPR训练结束后加载。")
def test_mpr_role_bias_is_centered_and_rescue_config():
    g=torch.Generator().manual_seed(112); model=MultiRolePrototypeClassifier(torch.randn(200,768,generator=g),torch.randn(200,3,768,generator=g),torch.tensor(10.0),learn_role_bias=True); model.raw_role_bias.data.copy_(torch.tensor([1.0,-1.0,0.5])); assert abs(float(model.role_bias().sum().detach()))<1e-7
    config,_=load_config(ROOT/"config/tries/v2_try_047_mpr_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-047"; assert config["learn_role_bias"] is True
