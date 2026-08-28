from pathlib import Path
import torch
from model.candidates.v2.modules.cnra import ClassNameResidualAlignment
from model.candidates.v2.trainers.train_cnra import load_config
ROOT=Path(__file__).resolve().parents[3]
def test_cnra_starts_off_and_beta_is_trainable():
    generator=torch.Generator().manual_seed(359); names=torch.randn(8,6,generator=generator); model=ClassNameResidualAlignment(names,5.0); parent=torch.randn(4,8,generator=generator); images=torch.randn(4,6,generator=generator); assert torch.equal(model(parent,images),parent); model(parent,images).sum().backward(); assert model.raw_beta.grad is not None
def test_cnra_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_138_cnra_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-138" and config["max_beta"]==5.0 and config["class_name_embeddings_sha256"]; source=(ROOT/"model/candidates/v2/trainers/train_cnra.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_cnra_reliability_configs_bind_jbec_parents():
    expected={"139":7,"140":27,"141":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_cnra_seed{seed}.yaml"); assert config["seed"]==seed and config["jbec_model_sha256"] and config["class_name_embeddings_sha256"]
