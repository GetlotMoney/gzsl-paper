from pathlib import Path
import torch
from model.innovations.adma import AttributeDiagonalMetric
from model.innovations.train_adma import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_adma_starts_as_attribute_cosine_and_is_trainable():
    generator=torch.Generator().manual_seed(331); predicted=torch.randn(5,12,generator=generator); classes=torch.randn(8,12,generator=generator); model=AttributeDiagonalMetric(12,0.1); expected=torch.nn.functional.normalize(predicted,dim=-1)@torch.nn.functional.normalize(classes,dim=-1).T; assert torch.equal(model.logits(predicted,classes),expected); model.logits(predicted,classes).sum().backward(); assert model.raw_log_weight.grad is not None
def test_adma_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_136_adma_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-136" and config["max_log_weight"]==0.1; source=(ROOT/"model/innovations/train_adma.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
