from pathlib import Path
import torch
import torch.nn.functional as F
from model.innovations.sdm import SymmetricDiagonalMetric
from model.innovations.train_sdm import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_sdm_starts_as_exact_cosine_and_is_bounded():
    generator=torch.Generator().manual_seed(211); images=torch.randn(6,768,generator=generator); prototypes=torch.randn(10,768,generator=generator); metric=SymmetricDiagonalMetric(max_log_weight=0.1); expected=F.normalize(images,dim=-1)@F.normalize(prototypes,dim=-1).T*12.0; assert torch.equal(metric.logits(images,prototypes,torch.tensor(12.0)),expected); metric.logits(images,prototypes,torch.tensor(12.0)).sum().backward(); assert metric.raw_log_weight.grad is not None; assert metric.weight().min()>=torch.exp(torch.tensor(-0.2)) and metric.weight().max()<=torch.exp(torch.tensor(0.2))
def test_sdm_config_and_training_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_086_sdm_seed17.yaml"); assert config["idea_id"]=="IDEA-028"; assert config["max_log_weight"]==0.1; source=(ROOT/"model/innovations/train_sdm.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
