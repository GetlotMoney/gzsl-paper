from pathlib import Path
from model.innovations.train_hgcs_episode import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_hgcs_episode_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_147_hgcs_rescue1_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-147" and config["group_count"]==20 and config["max_group_beta"]==10.0; source=(ROOT/"model/innovations/train_hgcs_episode.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source and 'fixed_class_folds' in source and '_fold_package' in source
