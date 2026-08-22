from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.ccgr import ClassConditionedGeometricGenerator,tangent_direction_basis
from model.innovations.train_ccgr import ccgr_class_features,load_config
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
def test_ccgr_magnitude_penalty_config():
    config,_=load_config(ROOT/"config/tries/v2_try_061_ccgr_rescue3_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-061"; assert config["magnitude_penalty"]==0.01
def test_ccgr_multiseed_configs_bind_own_ntr_parent():
    expected={"062":5,"063":6,"064":8,"065":9}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_ccgr_seed{seed}.yaml"); assert config["seed"]==seed; assert config["ntr_gate_model"]; assert config["fold_checkpoint_dir"]; assert config["pseudo_unseen_weight"]==0.25
def test_ccgr_magnitude_tune_configs():
    c15,_=load_config(ROOT/"config/tries/v2_try_066_ccgr_mag015_seed7.yaml"); c20,_=load_config(ROOT/"config/tries/v2_try_067_ccgr_mag020_seed7.yaml"); assert c15["max_magnitude"]==0.15; assert c20["max_magnitude"]==0.2
def test_eaml_margin_config():
    config,_=load_config(ROOT/"config/tries/v2_try_074_eaml_seed7.yaml"); assert config["idea_id"]=="IDEA-022"; assert config["pseudo_unseen_margin"]==0.1; assert config["max_magnitude"]==0.2
def test_ccgr_epoch_selection_config():
    config,_=load_config(ROOT/"config/tries/v2_try_077_ccgr_epoch_select_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-077"; assert config["select_each_epoch"] is True; assert config["max_magnitude"]==0.2
def test_ccgr_epoch_selection_training_seeds():
    expected={"078":17,"079":27,"080":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_ccgr_epoch_seed{seed}.yaml"); assert config["seed"]==seed; assert config["select_each_epoch"] is True
def test_ng_ccgr_top5_vector_features():
    g=torch.Generator().manual_seed(19); base=torch.randn(200,768,generator=g); value=torch.randn(200,768,generator=g); resultant=torch.rand(200,generator=g); top5=torch.rand(200,5,generator=g)
    mean_features=ccgr_class_features(base,value,resultant,top5,"top5_mean"); vector_features=ccgr_class_features(base,value,resultant,top5,"top5_vector")
    assert mean_features.shape==(200,4); assert vector_features.shape==(200,8); assert torch.equal(vector_features[:,:3],mean_features[:,:3]); assert torch.allclose(mean_features[:,3],top5.mean(dim=1))
    basis=tangent_direction_basis(base,value,torch.randn(200,3,768,generator=g)); model=ClassConditionedGeometricGenerator(base,basis,vector_features,torch.arange(150,200),torch.tensor(10.0)); assert model.trunk[0].in_features==8
def test_ng_ccgr_config():
    config,_=load_config(ROOT/"config/tries/v2_try_081_ng_ccgr_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-081"; assert config["idea_id"]=="IDEA-025"; assert config["class_feature_mode"]=="top5_vector"; assert config["select_each_epoch"] is True
