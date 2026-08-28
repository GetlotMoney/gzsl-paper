from pathlib import Path
import torch
from model.candidates.v2.modules.ara import AttributeResidualAlignment
from model.candidates.v2.modules.cara import ConfidenceAwareAttributeResidual
from model.candidates.v2.trainers.train_cara import load_config
ROOT=Path(__file__).resolve().parents[3]
def test_cara_starts_at_parent_and_has_variable_residual():
    generator=torch.Generator().manual_seed(263); ridge=torch.randn(8,6,generator=generator); attributes=torch.randn(5,6,generator=generator); base=AttributeResidualAlignment(ridge,attributes); base.raw_beta.data.fill_(0.5); model=ConfidenceAwareAttributeResidual(base); images=torch.randn(7,8,generator=generator); prototypes=torch.nn.functional.normalize(torch.randn(5,8,generator=generator),dim=-1); assert torch.equal(model.logits(images,prototypes,torch.tensor(10.)),model.logits(images,prototypes,torch.tensor(10.),enabled=False)); model.logits(images,prototypes,torch.tensor(10.)).sum().backward(); assert [name for name,p in model.named_parameters() if p.requires_grad]==["gate.0.weight","gate.0.bias","gate.2.weight","gate.2.bias"]
def test_cara_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_102_cara_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-102" and config["max_beta_residual"]==4.0; source=(ROOT/"model/candidates/v2/trainers/train_cara.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
