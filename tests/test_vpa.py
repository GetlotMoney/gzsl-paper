from pathlib import Path
import torch
from model.innovations.ara import AttributeResidualAlignment
from model.innovations.sdm import SymmetricDiagonalMetric
from model.innovations.vpa import VisualPrototypeAttributeResidual,fit_attribute_to_visual_map
from model.innovations.train_vpa import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_vpa_reverse_ridge_and_initial_off():
    generator=torch.Generator().manual_seed(307); attributes=torch.randn(8,5,generator=generator); seen=torch.arange(6); centroids=torch.randn(6,7,generator=generator); weight=fit_attribute_to_visual_map(attributes,seen,centroids,0.01); assert weight.shape==(5,7); forward=torch.randn(7,5,generator=generator); base=AttributeResidualAlignment(forward,attributes); visual=torch.nn.functional.normalize(attributes,dim=-1)@weight; model=VisualPrototypeAttributeResidual(base,visual); images=torch.randn(4,7,generator=generator); prototypes=torch.nn.functional.normalize(torch.randn(8,7,generator=generator),dim=-1); metric=SymmetricDiagonalMetric(dimension=7); assert torch.equal(model.logits(images,prototypes,torch.tensor(9.),metric),model.logits(images,prototypes,torch.tensor(9.),metric,enabled=False))
def test_vpa_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_118_vpa_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-118" and config["reverse_ridge"]==0.01; source=(ROOT/"model/innovations/train_vpa.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_vpa_reliability_configs_bind_cra_parents():
    expected={"122":7,"123":27,"124":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_vpa_seed{seed}.yaml"); assert config["seed"]==seed and config["cra_model_sha256"] and config["ccgr_model_sha256"]
def test_vpa_reverse_ridge_tune_configs():
    c01,_=load_config(ROOT/"config/tries/v2_try_128_vpa_reverse01_seed17.yaml"); c1,_=load_config(ROOT/"config/tries/v2_try_129_vpa_reverse1_seed17.yaml"); assert c01["reverse_ridge"]==0.1 and c1["reverse_ridge"]==1.0
