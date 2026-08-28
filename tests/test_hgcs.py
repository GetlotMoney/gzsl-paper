from pathlib import Path
import torch
from model.candidates.v2.modules.hgcs import HierarchicalGroupCommonSuppression,spherical_name_groups
from model.candidates.v2.trainers.train_hgcs import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_hgcs_groups_and_initial_off():
    generator=torch.Generator().manual_seed(367); names=torch.randn(30,8,generator=generator); centers,assignment=spherical_name_groups(names,5); assert centers.shape==(5,8) and assignment.shape==(30,); model=HierarchicalGroupCommonSuppression(names,5,10.0); parent=torch.randn(4,30,generator=generator); images=torch.randn(4,8,generator=generator); assert torch.equal(model(parent,images),parent); model(parent,images).sum().backward(); assert model.raw_beta.grad is not None
def test_hgcs_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_146_hgcs_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-146" and config["group_count"]==20; source=(ROOT/"model/candidates/v2/trainers/train_hgcs.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
