from pathlib import Path
from model.innovations.train_vebc import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_vebc_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_119_vebc_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-119" and config["max_gamma"]==0.25 and config["vpa_model_sha256"]; source=(ROOT/"model/innovations/train_vebc.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source and 'pseudo_seen_mask' in source
def test_vebc_fine_lr_rescue_config():
    config,_=load_config(ROOT/"config/tries/v2_try_120_vebc_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-120" and config["lr"]==0.0025
def test_vebc_interior_gamma_rescue_config():
    config,_=load_config(ROOT/"config/tries/v2_try_121_vebc_rescue2_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-121" and config["lr"]==0.0025 and config["max_gamma"]==0.3
def test_vebc_reliability_configs_bind_vpa_parents():
    expected={"125":7,"126":27,"127":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_vebc_seed{seed}.yaml"); assert config["seed"]==seed and config["max_gamma"]==0.3 and config["vpa_model_sha256"]
def test_vebc_reverse_ridge_tune_config():
    config,_=load_config(ROOT/"config/tries/v2_try_130_vebc_reverse01_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-130" and config["reverse_ridge"]==0.1 and config["vpa_model_sha256"]
