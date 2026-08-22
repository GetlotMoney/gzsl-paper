from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.ort import OrthogonalMix, orthogonal_transport, residual_subspace
from model.innovations.tst import tangent_transport
from model.innovations.train_ort import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_ort_zero_mix_matches_tst_and_basis_shape():
    g=torch.Generator().manual_seed(131); base=torch.randn(200,768,generator=g); source_classes=torch.arange(100); adapted=torch.randn(100,768,generator=g); basis=residual_subspace(base,adapted,source_classes,32); assert basis.shape==(32,768); target=torch.arange(100,150); b=base.index_select(0,target); value=torch.randn(50,768,generator=g); step=torch.rand(50,generator=g); assert torch.allclose(orthogonal_transport(b,value,step,basis,torch.tensor(0.0)),tangent_transport(b,value,step),atol=1e-6)
def test_ort_mix_trainable_and_config():
    mix=OrthogonalMix(); mix().square().backward(); assert mix.raw_mix.grad is not None; config,_=load_config(ROOT/"config/tries/v2_try_056_ort_seed7.yaml"); assert config["idea_id"]=="IDEA-017"; assert config["subspace_rank"]==32
def test_ort_complement_mode_and_rescue_config():
    g=torch.Generator().manual_seed(132); base=torch.randn(200,768,generator=g); source_classes=torch.arange(100); adapted=torch.randn(100,768,generator=g); basis=residual_subspace(base,adapted,source_classes,32); target=torch.arange(100,150); b=base.index_select(0,target); value=torch.randn(50,768,generator=g); step=torch.rand(50,generator=g); shared=orthogonal_transport(b,value,step,basis,torch.tensor(1.0),"shared"); complement=orthogonal_transport(b,value,step,basis,torch.tensor(1.0),"complement"); assert not torch.equal(shared,complement)
    config,_=load_config(ROOT/"config/tries/v2_try_057_ort_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-057"; assert config["projection_mode"]=="complement"
