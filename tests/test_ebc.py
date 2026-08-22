from pathlib import Path
import torch
from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.train_ebc import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_ebc_starts_off_and_only_changes_seen_logits():
    generator=torch.Generator().manual_seed(293); logits=torch.randn(6,10,generator=generator); seen=torch.tensor([True]*6+[False]*4); model=EpisodicBiasCalibration(0.2); assert torch.equal(model(logits,seen),logits); model.raw_gamma.data.fill_(0.5); changed=model(logits,seen); assert torch.equal(changed[:,~seen],logits[:,~seen]) and not torch.equal(changed[:,seen],logits[:,seen]); changed.sum().backward(); assert model.raw_gamma.grad is not None
def test_ebc_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_113_ebc_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-113" and config["max_gamma"]==0.2; source=(ROOT/"model/innovations/train_ebc.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_ebc_conservative_rescue_config():
    config,_=load_config(ROOT/"config/tries/v2_try_114_ebc_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-114" and config["max_gamma"]==0.15
