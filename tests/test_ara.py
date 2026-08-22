from pathlib import Path
import torch
from model.innovations.ara import AttributeResidualAlignment,fit_ridge_attribute_map
from model.innovations.sdm import SymmetricDiagonalMetric
from model.innovations.train_ara import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_ara_starts_at_parent_and_ridge_has_expected_shape():
    generator=torch.Generator().manual_seed(241); train=torch.randn(20,8,generator=generator); labels=torch.arange(20)%5; attributes=torch.randn(5,6,generator=generator); weight=fit_ridge_attribute_map(train,labels,attributes,0.01); assert weight.shape==(8,6); images=torch.randn(4,8,generator=generator); prototypes=torch.randn(5,8,generator=generator); metric=SymmetricDiagonalMetric(dimension=8); ara=AttributeResidualAlignment(weight,attributes,max_beta=20); assert torch.equal(ara.logits(images,prototypes,torch.tensor(10.),metric),metric.logits(images,prototypes,torch.tensor(10.))); ara.logits(images,prototypes,torch.tensor(10.),metric).sum().backward(); assert ara.raw_beta.grad is not None and abs(float(ara.beta().detach()))<1e-8
def test_ara_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_092_ara_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-092" and config["ridge"]==0.01; source=(ROOT/"model/innovations/train_ara.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_ara_reliability_configs_bind_own_parents():
    expected={"093":7,"094":27,"095":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_ara_seed{seed}.yaml"); assert config["seed"]==seed; assert config["ccgr_model_sha256"] and config["sdm_model_sha256"]
