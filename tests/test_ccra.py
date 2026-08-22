from pathlib import Path
import torch
from model.innovations.ara import AttributeResidualAlignment
from model.innovations.ccra import ClassConditionedAttributeResidual,attribute_pca_features
from model.innovations.train_ccra import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_ccra_starts_at_parent_and_has_class_features():
    generator=torch.Generator().manual_seed(281); attributes=torch.randn(20,12,generator=generator); features=attribute_pca_features(attributes,6); assert features.shape==(20,6); ridge=torch.randn(10,12,generator=generator); base=AttributeResidualAlignment(ridge,attributes); base.raw_beta.data.fill_(0.4); model=ClassConditionedAttributeResidual(base,features); images=torch.randn(5,10,generator=generator); prototypes=torch.nn.functional.normalize(torch.randn(20,10,generator=generator),dim=-1); assert torch.equal(model.logits(images,prototypes,torch.tensor(8.)),model.logits(images,prototypes,torch.tensor(8.),enabled=False)); model.logits(images,prototypes,torch.tensor(8.)).sum().backward(); assert model.gate[-1].weight.grad is not None
def test_ccra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_108_ccra_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-108" and config["attribute_rank"]==16; source=(ROOT/"model/innovations/train_ccra.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
