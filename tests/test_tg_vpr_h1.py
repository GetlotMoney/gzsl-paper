from __future__ import annotations

from pathlib import Path
import unittest

import torch
import yaml

from model.frameworks.v2 import TGVPRH1FixedEqual
from model.frameworks.v2 import train as h1_train
from tools.runtime import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def make_model(inner_ratio: float = 0.35, dropout: float = 0.0):
    generator = torch.Generator().manual_seed(2608)
    sentences = torch.randn(200, 8, 768, generator=generator)
    classes = torch.tensor([class_id for class_id in range(200) if class_id % 4 != 0])
    centroids = torch.randn(150, 768, generator=generator)
    return TGVPRH1FixedEqual(
        sentences,
        classes,
        centroids,
        dropout=dropout,
        inner_ratio=inner_ratio,
        outer_ratio=0.65,
        temperature=0.07,
    )


def test_groups_shapes_and_unseen_mean8_boundary():
    model = make_model().eval()
    groups = model.semantic_group_vectors()
    prototypes = model.prototypes()
    unseen = torch.arange(200)[~torch.isin(torch.arange(200), model.adapted_classes)]
    assert groups.shape == (200, 3, 768)
    assert prototypes.shape == (200, 768)
    assert torch.allclose(groups.norm(dim=-1), torch.ones(200, 3), atol=1e-6)
    assert torch.equal(
        prototypes.index_select(0, unseen),
        model.base_prototypes().index_select(0, unseen),
    )


def test_fixed_equal_weights_block_group_logit_gradient_but_train_value():
    model = make_model()
    images = torch.randn(6, 768, generator=torch.Generator().manual_seed(27))
    loss = torch.nn.functional.cross_entropy(
        model.logits(images, model.adapted_classes), torch.arange(6)
    )
    loss = loss + 0.1 * model.topology_loss()
    loss.backward()
    assert torch.equal(model.semantic_group_weights(), torch.full((3,), 1.0 / 3.0))
    assert model.semantic_group_logits.grad is None
    assert float(model.tg_value_projection.weight.grad.abs().sum()) > 0


def test_inner_residual_changes_seen_only_and_components_reconstruct_logits():
    low = make_model(inner_ratio=0.20)
    high = make_model(inner_ratio=0.65)
    high.load_state_dict(low.state_dict(), strict=True)
    low.eval()
    high.eval()
    unseen = torch.arange(200)[~torch.isin(torch.arange(200), low.adapted_classes)]
    assert not torch.allclose(
        low.prototypes().index_select(0, low.adapted_classes),
        high.prototypes().index_select(0, high.adapted_classes),
    )
    assert torch.equal(
        low.prototypes().index_select(0, unseen),
        high.prototypes().index_select(0, unseen),
    )
    images = torch.randn(4, 768, generator=torch.Generator().manual_seed(47))
    base, roles = low.logit_components(images)
    assert torch.allclose(base + roles.sum(dim=-1), low.logits(images), atol=2e-5)


def test_formal_config_and_protocol_contract():
    config_path = ROOT / "config" / "tg_vpr_h1.yaml"
    config, digest = h1_train.load_config(config_path)
    assert digest == sha256_file(config_path)
    assert config["framework_id"] == "FRAMEWORK-V2"
    assert config["module_id"] == "INNOVATION-MODULE-1"
    assert config["evaluation_protocol"] == "test_selected_inductive_gzsl"
    assert config["test_used_for_selection"] is True
    assert config["unseen_images_used_for_gradient"] is False
    assert config["inner_ratio"] == 0.35
    assert config["outer_ratio"] == 0.65
    assert config["topology_weight"] == 0.1
    assert config["group_weights"] == [1.0 / 3.0] * 3
    assert config["value_heads"] == 1


def test_source_manifest_is_clean_snapshot_not_old_experiment_tree():
    source = yaml.safe_load(
        (ROOT / "docs" / "TG_VPR_H1_SOURCE.yaml").read_text(encoding="utf-8")
    )
    assert source["module_id"] == "INNOVATION-MODULE-1"
    assert source["target_framework"] == "FRAMEWORK-V2"
    assert source["source_result_commit"] == (
        "0aa38ab46020690879c1de8b937f35fd6b607f22"
    )
    assert source["framework_v1_integration"] == "not_applied"
    assert source["framework_v2_status"] == "baseline_completed_single_seed"
    for record in source["target_files"].values():
        assert sha256_file(ROOT / record["path"]) == record["sha256"]
    assert not (ROOT / "experiments" / "v5").exists()
    assert not (ROOT / "research" / "ideas" / "IDEA-0021_tg_vpr_h1").exists()


class TGVPRH1Test(unittest.TestCase):
    def test_shapes(self):
        test_groups_shapes_and_unseen_mean8_boundary()

    def test_gradients(self):
        test_fixed_equal_weights_block_group_logit_gradient_but_train_value()

    def test_residual(self):
        test_inner_residual_changes_seen_only_and_components_reconstruct_logits()

    def test_config(self):
        test_formal_config_and_protocol_contract()

    def test_source(self):
        test_source_manifest_is_clean_snapshot_not_old_experiment_tree()


if __name__ == "__main__":
    unittest.main()
