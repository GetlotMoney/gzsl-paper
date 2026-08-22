from pathlib import Path
import torch
from model.innovations.dra import DescriptionResidualAlignment,fit_ridge_description_map
from model.innovations.train_dra import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_dra_starts_at_parent_and_role_ridge_shape():
    generator=torch.Generator().manual_seed(251); train=torch.randn(20,8,generator=generator); labels=torch.arange(20)%5; descriptions=torch.randn(5,3,4,generator=generator); weight=fit_ridge_description_map(train,labels,descriptions,0.01); assert weight.shape==(8,12); images=torch.randn(4,8,generator=generator); prototypes=torch.randn(5,8,generator=generator); dra=DescriptionResidualAlignment(weight,descriptions); parent=torch.nn.functional.normalize(images,dim=-1)@prototypes.T*10; assert torch.equal(dra.logits(images,prototypes,torch.tensor(10.)),parent); dra.logits(images,prototypes,torch.tensor(10.)).sum().backward(); assert dra.raw_beta.grad is not None
def test_dra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_101_dra_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-101" and config["ridge"]==0.01; source=(ROOT/"model/innovations/train_dra.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
