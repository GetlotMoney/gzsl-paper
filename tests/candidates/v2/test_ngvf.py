from pathlib import Path
import torch
from model.candidates.v2.modules.ngvf import NormalizedGeometricVisualFusion
from model.candidates.v2.trainers.train_ngvf import load_config
ROOT=Path(__file__).resolve().parents[3]
def test_ngvf_starts_at_additive_and_trains_eta():
    generator=torch.Generator().manual_seed(347); additive=torch.randn(5,8,generator=generator); normalized=torch.randn(5,8,generator=generator); model=NormalizedGeometricVisualFusion(); assert torch.equal(model(additive,normalized),additive); model(additive,normalized).sum().backward(); assert model.raw_eta.grad is not None
def test_ngvf_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_137_ngvf_seed17.yaml"); assert config["attempt_id"]=="V2-TRY-137" and config["reverse_ridge"]==0.01; source=(ROOT/"model/candidates/v2/trainers/train_ngvf.py").read_text(encoding="utf-8"); assert 'unseen_images_used_for_gradient":False' in source
