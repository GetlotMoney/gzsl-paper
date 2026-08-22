from pathlib import Path
import torch
from model.innovations.jbec import JointBidirectionalEpisodicCalibration
from model.innovations.train_jbec import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_jbec_starts_at_parent_and_trains_two_residuals():
    generator=torch.Generator().manual_seed(317); cra=torch.randn(5,8,generator=generator); visual=torch.randn(5,8,generator=generator); mask=torch.tensor([True]*4+[False]*4); model=JointBidirectionalEpisodicCalibration(10.0,0.25); assert torch.equal(model(cra,visual,mask),model(cra,visual,mask,enabled=False)); model(cra,visual,mask).sum().backward(); assert model.raw_beta_residual.grad is not None and model.raw_gamma_residual.grad is not None
def test_jbec_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_131_jbec_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-131" and config["max_beta_residual"]==2.0 and config["max_gamma_residual"]==0.05; source=(ROOT/"model/innovations/train_jbec.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
def test_jbec_reliability_configs_bind_vebc_parents():
    expected={"132":7,"133":27,"134":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_jbec_seed{seed}.yaml"); assert config["seed"]==seed and config["vebc_model_sha256"] and config["vpa_model_sha256"]
def test_jbec_gamma_residual_tune_config():
    config,_=load_config(ROOT/"config/tries/v2_try_135_jbec_gamma01_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-135" and config["max_gamma_residual"]==0.1
