from pathlib import Path
from model.candidates.v2.trainers.train_cnebc import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_cnebc_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_142_cnebc_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-142" and config["max_gamma_residual"]==0.1 and config["cnra_model_sha256"]; source=(ROOT/"model/candidates/v2/trainers/train_cnebc.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source and 'pseudo_seen_mask' in source
def test_cnebc_reliability_configs_bind_cnra_parents():
    expected={"143":7,"144":27,"145":37}
    for suffix,seed in expected.items():
        config,_=load_config(ROOT/f"config/tries/v2_try_{suffix}_cnebc_seed{seed}.yaml"); assert config["seed"]==seed and config["cnra_model_sha256"] and config["jbec_model_sha256"]
