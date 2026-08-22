from pathlib import Path
import torch
from model.innovations.cnra import ClassNameResidualAlignment
from model.innovations.train_cnra import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_cnra_starts_off_and_beta_is_trainable():
    generator=torch.Generator().manual_seed(359); names=torch.randn(8,6,generator=generator); model=ClassNameResidualAlignment(names,5.0); parent=torch.randn(4,8,generator=generator); images=torch.randn(4,6,generator=generator); assert torch.equal(model(parent,images),parent); model(parent,images).sum().backward(); assert model.raw_beta.grad is not None
def test_cnra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_138_cnra_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-138" and config["max_beta"]==5.0 and config["class_name_embeddings_sha256"]; source=(ROOT/"model/innovations/train_cnra.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
