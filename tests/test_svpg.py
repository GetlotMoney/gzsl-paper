from __future__ import annotations
from pathlib import Path
import torch
from model.innovations.svpg import SemanticVisualPrototypeGenerator
from model.innovations.train_svpg import load_config
ROOT=Path(__file__).resolve().parents[1]
def test_svpg_starts_as_parent_and_trains_shared_adapter():
    g=torch.Generator().manual_seed(121); parent=torch.randn(200,768,generator=g); model=SemanticVisualPrototypeGenerator(parent,torch.tensor(10.0)); assert torch.allclose(model.prototypes(),model.prototypes(enabled=False),atol=1e-7); images=torch.randn(6,768,generator=g); torch.nn.functional.cross_entropy(model.logits(images),torch.arange(6)).backward(); assert any(p.grad is not None for p in model.adapter.parameters())
def test_svpg_config_and_boundary():
    config,_=load_config(ROOT/"config/tries/v2_try_053_svpg_seed7.yaml"); assert config["idea_id"]=="IDEA-016"; source=(ROOT/"model/innovations/train_svpg.py").read_text(encoding="utf-8"); assert source.index("for epoch in range")<source.index("# official test严格在SVPG训练结束后加载。")
def test_svpg_unseen_only_scope_and_rescue_config():
    g=torch.Generator().manual_seed(122); parent=torch.randn(200,768,generator=g); unseen=torch.arange(150,200); model=SemanticVisualPrototypeGenerator(parent,torch.tensor(10.0),target_classes=unseen); model.adapter[-1].bias.data.fill_(0.1); changed=model.prototypes(); base=model.prototypes(enabled=False); assert torch.equal(changed[:150],base[:150]); assert not torch.equal(changed[150:],base[150:])
    config,_=load_config(ROOT/"config/tries/v2_try_054_svpg_rescue1_seed7.yaml"); assert config["attempt_id"]=="V2-TRY-054"; assert config["apply_scope"]=="unseen_only"
