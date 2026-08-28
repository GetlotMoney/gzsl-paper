from pathlib import Path
import torch
from model.candidates.v2.modules.sfa import SemanticFactorAlignment,fit_ridge_factor_map,semantic_factor_matrix
from model.candidates.v2.trainers.train_sfa import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_sfa_factors_are_low_rank_and_initially_off():
    generator=torch.Generator().manual_seed(271); descriptions=torch.randn(20,3,8,generator=generator); factors=semantic_factor_matrix(descriptions,6); assert factors.shape==(20,6) and torch.allclose(factors.norm(dim=1),torch.ones(20)); train=torch.randn(40,10,generator=generator); labels=torch.arange(40)%20; weight=fit_ridge_factor_map(train,labels,factors,0.01); model=SemanticFactorAlignment(weight,factors); images=torch.randn(5,10,generator=generator); prototypes=torch.nn.functional.normalize(torch.randn(20,10,generator=generator),dim=-1); parent=torch.nn.functional.normalize(images,dim=-1)@prototypes.T*9; assert torch.equal(model.logits(images,prototypes,torch.tensor(9.)),parent)
def test_sfa_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_103_sfa_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-103" and config["factor_rank"]==64; source=(ROOT/"model/candidates/v2/trainers/train_sfa.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
