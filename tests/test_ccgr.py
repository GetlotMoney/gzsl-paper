from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.ccgr import ClassConditionedGeometricGenerator,NeighborhoodResidualCCGR,tangent_direction_basis
from model.innovations.train_ccgr import ccgr_class_features,harmonic_episode_loss,local_boundary_loss,load_config
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
def test_neighborhood_residual_starts_exactly_at_parent():
    g=torch.Generator().manual_seed(29); base=torch.randn(200,768,generator=g); value=torch.randn(200,768,generator=g); roles=torch.randn(200,3,768,generator=g); top5=torch.rand(200,5,generator=g); resultant=torch.rand(200,generator=g); vector=ccgr_class_features(base,value,resultant,top5,"top5_vector"); mean=ccgr_class_features(base,value,resultant,top5,"top5_mean"); basis=tangent_direction_basis(base,value,roles); unseen=torch.arange(150,200); parent=ClassConditionedGeometricGenerator(base,basis,mean,unseen,torch.tensor(10.0),max_magnitude=0.2)
    with torch.no_grad():
        for parameter in parent.parameters(): parameter.normal_(0.0,0.05)
    residual=NeighborhoodResidualCCGR(base,basis,vector,unseen,torch.tensor(10.0),parent.state_dict()); assert torch.equal(parent.generated_all(),residual.generated_all()); assert [name for name,p in residual.named_parameters() if p.requires_grad]==["neighborhood_residual.weight"]
def test_neighborhood_residual_config():
    config,_=load_config(ROOT/"config/tries/v2_try_082_ng_ccgr_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-082"; assert config["parent_ccgr_model_sha256"]; assert config["class_feature_mode"]=="top5_vector"
def test_harmonic_episode_loss_balances_both_groups():
    logits=torch.tensor([[2.0,0.0],[0.0,2.0],[1.0,0.0],[0.0,1.0]],requires_grad=True); targets=torch.tensor([0,1,0,1]); loss,seen,unseen=harmonic_episode_loss(logits,targets,2); assert 0<loss<1; assert seen>unseen; loss.backward(); assert logits.grad is not None and torch.isfinite(logits.grad).all()
def test_ccgr_heo_config():
    config,_=load_config(ROOT/"config/tries/v2_try_083_ccgr_heo_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-083"; assert config["idea_id"]=="IDEA-026"; assert config["harmonic_weight"]==1.0; assert config["parent_ccgr_model_sha256"]
def test_ccgr_heo_conservative_config():
    config,_=load_config(ROOT/"config/tries/v2_try_084_ccgr_heo_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-084"; assert config["idea_id"]=="IDEA-026"; assert config["harmonic_weight"]==0.1
def test_local_boundary_loss_targets_hardest_wrong_class():
    generated=torch.eye(4); centroids=torch.stack((torch.tensor([0.8,0.6,0.,0.]),torch.tensor([0.7,0.714,0.,0.]))); loss=local_boundary_loss(generated,centroids,torch.tensor([0,1]),0.02); assert loss>0; loss.backward() if loss.requires_grad else None
def test_ccgr_lbs_config():
    config,_=load_config(ROOT/"config/tries/v2_try_085_ccgr_lbs_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-085"; assert config["idea_id"]=="IDEA-027"; assert config["hard_negative_weight"]==0.1; assert config["hard_negative_margin"]==0.02
