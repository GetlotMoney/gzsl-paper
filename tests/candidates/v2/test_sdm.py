from pathlib import Path
import torch
import torch.nn.functional as F
from model.candidates.v2.modules.sdm import SymmetricDiagonalMetric,SymmetricLowRankMetric
from model.candidates.v2.trainers.train_sdm import load_config,principal_centroid_basis
ROOT=Path(__file__).resolve().parents[3]
def test_sdm_starts_as_exact_cosine_and_is_bounded():
    generator=torch.Generator().manual_seed(211); images=torch.randn(6,768,generator=generator); prototypes=torch.randn(10,768,generator=generator); metric=SymmetricDiagonalMetric(max_log_weight=0.1); expected=F.normalize(images,dim=-1)@F.normalize(prototypes,dim=-1).T*12.0; assert torch.equal(metric.logits(images,prototypes,torch.tensor(12.0)),expected); metric.logits(images,prototypes,torch.tensor(12.0)).sum().backward(); assert metric.raw_log_weight.grad is not None; assert metric.weight().min()>=torch.exp(torch.tensor(-0.2)) and metric.weight().max()<=torch.exp(torch.tensor(0.2))
def test_sdm_config_and_training_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_086_sdm_seed17.yaml"); assert config["idea_id"]=="IDEA-028"; assert config["max_log_weight"]==0.1; source=(ROOT/"model/candidates/v2/trainers/train_sdm.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_low_rank_sdm_starts_at_parent_and_only_trains_subspace():
    generator=torch.Generator().manual_seed(223); images=torch.randn(8,768,generator=generator); prototypes=torch.randn(12,768,generator=generator); base=SymmetricDiagonalMetric(); base.raw_log_weight.data.normal_(0,0.1); basis=principal_centroid_basis(torch.randn(150,768,generator=generator),64); model=SymmetricLowRankMetric(base,basis); assert torch.equal(model.logits(images,prototypes,torch.tensor(9.0)),base.logits(images,prototypes,torch.tensor(9.0))); assert [name for name,p in model.named_parameters() if p.requires_grad]==["raw_subspace_log_weight"]
def test_low_rank_sdm_config():
    config,_=load_config(ROOT/"config/tries/v2_try_087_sdm_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-087"; assert config["subspace_rank"]==64; assert config["parent_sdm_model_sha256"]
def test_joint_low_rank_sdm_config_and_trainable_base():
    config,_=load_config(ROOT/"config/tries/v2_try_088_sdm_rescue2_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-088" and config["train_base_metric"] is True; base=SymmetricDiagonalMetric(); model=SymmetricLowRankMetric(base,torch.eye(64,768),freeze_base_metric=False); assert {name for name,p in model.named_parameters() if p.requires_grad}=={"raw_subspace_log_weight","base_metric.raw_log_weight"}
def test_sdm_reliability_configs_bind_own_parent():
    expected={"089":7,"090":27,"091":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_sdm_seed{seed}.yaml"); assert config["seed"]==seed; assert config["ccgr_model_sha256"]; assert config["schema_version"]=="gzsl-paper.sdm.v1"
