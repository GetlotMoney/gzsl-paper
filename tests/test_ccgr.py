from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.ccgr import ClassConditionedGeometricGenerator,tangent_direction_basis
from model.innovations.train_ccgr import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_ccgr_basis_and_unseen_only_application():
    g=torch.Generator().manual_seed(141); base=torch.randn(200,768,generator=g); value=torch.randn(200,768,generator=g); roles=torch.randn(200,3,768,generator=g); basis=tangent_direction_basis(base,value,roles); assert basis.shape==(200,4,768); features=torch.randn(200,4,generator=g,requires_grad=True); unseen=torch.arange(150,200); model=ClassConditionedGeometricGenerator(base,basis,features,unseen,torch.tensor(10.0)); changed=model.prototypes(); parent=model.prototypes(enabled=False); assert torch.equal(changed[:150],parent[:150]); assert not torch.equal(changed[150:],parent[150:]); optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    for _ in range(2):
        loss=model.generated_all().square().mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert model.feature_mean.grad_fn is None and model.feature_std.grad_fn is None
def test_ccgr_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_058_ccgr_seed7.yaml"); assert config["idea_id"]=="IDEA-018"; source=(ROOT/"model/innovations/train_ccgr.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在CCGR训练结束后加载。")
def test_ccgr_episode_config():
    config,_=load_config(ROOT/"config/tries/v2_try_059_ccgr_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-059"; assert config["training_objective"]=="pseudo_unseen_episode"; assert config["fold_checkpoint_dir"]
def test_ccgr_unseen_risk_config():
    config,_=load_config(ROOT/"config/tries/v2_try_060_ccgr_rescue2_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-060"; assert config["pseudo_unseen_weight"]==0.25
