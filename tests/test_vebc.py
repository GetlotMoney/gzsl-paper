from pathlib import Path
from model.innovations.train_vebc import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_vebc_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_119_vebc_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-119" and config["max_gamma"]==0.25 and config["vpa_model_sha256"]; source=(ROOT/"model/innovations/train_vebc.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source and 'pseudo_seen_mask' in source
