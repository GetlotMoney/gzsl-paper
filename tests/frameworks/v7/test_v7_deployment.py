from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from model.frameworks.v7.model import V7DeploymentModel, load_v7_checkpoint


def _export() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "q": torch.randn(200, 1536, generator=generator),
        "bias": torch.randn(200, generator=generator),
        "reader_in_weight": torch.randn(64, 768, generator=generator) * 0.01,
        "reader_in_bias": torch.randn(64, generator=generator) * 0.01,
        "reader_out_weight": torch.randn(768, 64, generator=generator) * 0.01,
        "reader_out_bias": torch.randn(768, generator=generator) * 0.01,
    }


def test_v7_forward_is_exact_exported_hq_plus_bias() -> None:
    export = _export()
    model = V7DeploymentModel.from_export(export)
    images = torch.randn(5, 768, generator=torch.Generator().manual_seed(8))
    hidden = F.linear(images, export["reader_in_weight"], export["reader_in_bias"])
    residual = F.linear(
        F.gelu(hidden), export["reader_out_weight"], export["reader_out_bias"]
    )
    h = torch.cat((F.normalize(images, dim=-1), F.normalize(images + residual, dim=-1)), dim=1)
    expected = h @ export["q"].T + export["bias"]
    actual = model(images)
    assert tuple(actual.shape) == (5, 200)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


def test_v7_checkpoint_identity_and_graph_free_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model_best.pth"
    torch.save(
        {
            "experiment_id": "V6-TRY-006",
            "code_commit": "8de7cebda0235ab12e1b4b8f669134c8f4e2c075",
            "config_sha256": "73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b",
            "export": _export(),
        },
        checkpoint_path,
    )
    model, checkpoint = load_v7_checkpoint(checkpoint_path)
    assert checkpoint["experiment_id"] == "V6-TRY-006"
    assert set(model.state_dict()) == {
        "q",
        "bias",
        "reader_in_weight",
        "reader_in_bias",
        "reader_out_weight",
        "reader_out_bias",
    }
    forbidden = {"edge_index", "relation_embeddings", "incidence", "laplacian_map"}
    assert forbidden.isdisjoint(model.state_dict())


def test_v7_rejects_incomplete_export() -> None:
    export = _export()
    export.pop("bias")
    try:
        V7DeploymentModel.from_export(export)
    except ValueError as error:
        assert "export字段" in str(error)
    else:
        raise AssertionError("不完整export必须被拒绝")
