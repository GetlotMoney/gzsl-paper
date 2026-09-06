from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import subprocess
import tempfile
import unittest

import torch
import yaml

from model.frameworks.v1.model import GTPJ
from tools.cub_data import validate_standard_class_counts, validate_standard_split_sizes
from tools.evaluation import evaluate_cached, test_cache_paths
from tools.run_contract import (
    REPO_ROOT,
    atomic_write_json,
    current_code_commit,
    is_new_best,
    materialize_best_model,
    prepare_output_dir,
    require_finite_gradients,
    require_finite_tensor_tree,
    save_epoch_artifacts,
    snapshot_state_dict,
    validate_best_metrics_identity,
)


ROOT = Path(__file__).resolve().parents[3]


def _config():
    return SimpleNamespace(
        num_class=6,
        dim_f_clip=16,
        pse_heads=2,
        pse_dropout=0.0,
        pse_inner_ratio=0.35,
        pse_outer_ratio=0.65,
        tf_common_dim=8,
        tf_heads=2,
        tf_dropout=0.0,
        weight_s2v=0.5,
        local_weight=0.2,
        score_mode="add",
        fgvd_select_k=4,
        icsa_hidden=8,
        icsa_ratio=0.008,
        sgmp_topk=1,
        sgmp_hidden=8,
        sgmp_neg_margin=0.2,
        lambda_consist=0.05,
        consist_temp=2.0,
        consist_dynamic_gamma=0.1,
        lambda_topo_pearson=0.1,
        lambda_bmdd=0.05,
        msdn_temp=2.0,
        lambda_mpp=0.05,
        lambda_neg=0.01,
    )


def test_model_smoke_and_logits_shape():
    torch.manual_seed(5)
    seen = torch.tensor([0, 2, 3, 5])
    unseen = torch.tensor([1, 4])
    model = GTPJ(
        _config(),
        seen,
        unseen,
        torch.randn(4, 16),
        torch.randn(2, 16),
        seen_sentence_embeds=torch.randn(4, 7, 16),
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 577, 16), is_train=False)
    assert output["clip_S_pp"].shape == (2, 6)
    assert torch.isfinite(output["clip_S_pp"]).all()


def test_model_backward_smoke():
    torch.manual_seed(5)
    seen = torch.tensor([0, 2, 3, 5])
    unseen = torch.tensor([1, 4])
    model = GTPJ(
        _config(),
        seen,
        unseen,
        torch.randn(4, 16),
        torch.randn(2, 16),
        seen_sentence_embeds=torch.randn(4, 7, 16),
    ).train()
    output = model(torch.randn(2, 577, 16), is_train=True)
    losses = model.compute_loss(dict(output, batch_label=torch.tensor([0, 2])))
    assert losses["loss"].ndim == 0
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_model_accepts_cuda_class_identity():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    seen = torch.tensor([0, 2, 3, 5], device=device)
    unseen = torch.tensor([1, 4], device=device)
    model = GTPJ(
        _config(),
        seen,
        unseen,
        torch.randn(4, 16, device=device),
        torch.randn(2, 16, device=device),
        seen_sentence_embeds=torch.randn(4, 7, 16, device=device),
    )
    assert torch.equal(model.seenclass, seen)
    assert torch.equal(model.unseenclass, unseen)


class _ControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nclass = 4
        self.register_buffer("seenclass", torch.tensor([0, 2]))
        self.register_buffer("unseenclass", torch.tensor([1, 3]))

    def forward(self, features, is_train=False):
        del is_train
        rows = {
            0: [10.0, 0.0, 0.0, 0.0],
            1: [0.0, 0.0, 10.0, 0.0],
            2: [10.0, 9.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0, 10.0],
        }
        ids = features[:, 0, 0].long().tolist()
        return {"clip_S_pp": torch.tensor([rows[i] for i in ids], device=features.device)}


class _NonFiniteModel(_ControlledModel):
    def forward(self, features, is_train=False):
        output = super().forward(features, is_train=is_train)
        output["clip_S_pp"][0, 0] = float("nan")
        return output


def test_gzsl_metric_semantics():
    cache = {
        "seen_cls": torch.tensor([[0.0], [1.0]]),
        "seen_patches": torch.zeros(2, 576, 1),
        "seen_labels": torch.tensor([0, 2]),
        "unseen_cls": torch.tensor([[2.0], [3.0]]),
        "unseen_patches": torch.zeros(2, 576, 1),
        "unseen_labels": torch.tensor([1, 3]),
    }
    seen, unseen, harmonic, zsl = evaluate_cached(
        _ControlledModel(),
        "cpu",
        cache,
        seenclasses=torch.tensor([0, 2]),
        unseenclasses=torch.tensor([1, 3]),
        batch_size=2,
    )
    assert seen == 1.0
    assert unseen == 0.5
    assert abs(harmonic - 2.0 / 3.0) < 1e-12
    assert zsl == 1.0
    try:
        evaluate_cached(
            _NonFiniteModel(),
            "cpu",
            cache,
            seenclasses=torch.tensor([0, 2]),
            unseenclasses=torch.tensor([1, 3]),
            batch_size=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("NaN logits 必须被拒绝")


def test_config_and_protocol_identity():
    raw = yaml.safe_load((ROOT / "config" / "v1.yaml").read_text(encoding="utf-8"))
    assert raw["dataset"]["value"] == "CUB"
    assert raw["num_class"]["value"] == 200
    source = (ROOT / "model" / "frameworks" / "v1" / "train.py").read_text(encoding="utf-8")
    assert 'FRAMEWORK_ID = "FRAMEWORK-V1"' in source
    assert 'EVALUATION_PROTOCOL = "test_selected_inductive_gzsl"' in source
    assert re.search(r'"--output-dir"[\s\S]+required=True', source)
    assert "evaluate_cached(" in source
    assert '"test_used_for_selection": True' in source
    assert '"unseen_images_used_for_gradient": False' in source
    assert "model/v5-template-v2" not in source


def test_repository_has_no_template_layer_or_legacy_experiment_tree():
    assert not (ROOT / "TEMPLATE.yaml").exists()
    assert not any(ROOT.glob("**/*template-v*"))
    framework = yaml.safe_load(
        (ROOT / "experiments" / "v1" / "FRAMEWORK.yaml").read_text(encoding="utf-8")
    )
    assert framework["framework_branch"] == "framework/v1"
    assert framework["framework_tag"] == "v1"
    promoted_v5 = yaml.safe_load(
        (ROOT / "experiments" / "v5" / "FRAMEWORK.yaml").read_text(encoding="utf-8")
    )
    assert promoted_v5["framework_branch"] == "framework/v5"
    assert promoted_v5["framework_tag"] == "v5"


def test_git_identity_is_bound_to_script_repository():
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    original = Path.cwd()
    try:
        os.chdir(REPO_ROOT.parent)
        assert current_code_commit() == expected
        assert test_cache_paths()["seen_cls"].parent == REPO_ROOT / "data" / "cache"
    finally:
        os.chdir(original)


def test_output_boundary_and_epoch_artifact_semantics():
    assert is_new_best(0.0, None)
    assert not is_new_best(0.0, 0.0)
    try:
        is_new_best(float("nan"), None)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN H 必须被拒绝")
    try:
        prepare_output_dir(REPO_ROOT / "forbidden-run-output")
    except ValueError:
        pass
    else:
        raise AssertionError("仓库内 output-dir 必须被拒绝")
    with tempfile.TemporaryDirectory(prefix="gzsl-paper-contract-") as temporary:
        output_dir = prepare_output_dir(Path(temporary).resolve() / "RUN-001")
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        best_state = snapshot_state_dict(model)
        save_epoch_artifacts(
            output_dir=output_dir,
            model=model,
            checkpoint={
                "epoch": 1,
                "best_H": 1.0,
                "best_metrics": {"U": 1.0, "S": 1.0, "H": 1.0, "ZS": 1.0, "epoch": 1},
                "best_model_state_dict": best_state,
                "model_state_dict": snapshot_state_dict(model),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
            },
            new_best=True,
        )
        with torch.no_grad():
            model.weight.fill_(2.0)
        save_epoch_artifacts(
            output_dir=output_dir,
            model=model,
            checkpoint={
                "epoch": 2,
                "best_H": 1.0,
                "best_metrics": {"U": 1.0, "S": 1.0, "H": 1.0, "ZS": 1.0, "epoch": 1},
                "best_model_state_dict": best_state,
                "model_state_dict": snapshot_state_dict(model),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
            },
            new_best=False,
        )
        best = torch.load(output_dir / "model_best.pth", weights_only=True)
        last = torch.load(output_dir / "checkpoint_last.pth", weights_only=False)
        assert best["weight"].item() == 1.0
        assert last["epoch"] == 2
        assert last["model_state_dict"]["weight"].item() == 2.0
        torch.save({"weight": torch.tensor([[9.0]])}, output_dir / "model_best.pth")
        materialize_best_model(
            output_dir=output_dir,
            model=model,
            best_state_dict=last["best_model_state_dict"],
        )
        restored = torch.load(output_dir / "model_best.pth", weights_only=True)
        assert restored["weight"].item() == 1.0
        atomic_write_json(output_dir / "metrics.json", {"H": 1.0})
        assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")) == {
            "H": 1.0
        }

        with torch.no_grad():
            model.weight.fill_(float("inf"))
        try:
            snapshot_state_dict(model)
        except ValueError:
            pass
        else:
            raise AssertionError("非有限模型状态必须被拒绝")

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        gradient_model = torch.nn.ParameterList([parameter])
        try:
            require_finite_gradients(gradient_model)
        except ValueError:
            pass
        else:
            raise AssertionError("非有限梯度必须被拒绝")

        for value in (float("nan"), float("inf"), complex(1.0, float("nan"))):
            try:
                require_finite_tensor_tree({"scheduler": {"eta_min": value}}, "state")
            except ValueError:
                pass
            else:
                raise AssertionError("非有限 Python 标量必须被拒绝")

        try:
            validate_best_metrics_identity(
                1.0,
                {"U": 1.0, "S": 1.0, "H": 1.0, "ZS": 1.0, "epoch": 3},
                checkpoint_epoch=2,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("未来 best epoch 必须被拒绝")

        try:
            validate_best_metrics_identity(
                1.0,
                {"U": 1.0, "S": 1.0, "H": 1.0, "ZS": 1.0, "epoch": 1},
                checkpoint_epoch=2,
                new_best=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("new_best 与当前 epoch 不一致必须被拒绝")


def test_standard_cub_class_counts():
    validate_standard_class_counts(torch.arange(150), torch.arange(50))
    try:
        validate_standard_class_counts(torch.arange(151), torch.arange(49))
    except ValueError:
        pass
    else:
        raise AssertionError("151/49 类划分必须被拒绝")
    validate_standard_split_sizes(7057, 1764, 2967)
    try:
        validate_standard_split_sizes(7056, 1764, 2967)
    except ValueError:
        pass
    else:
        raise AssertionError("非标准样本数必须被拒绝")


class V1ContractTest(unittest.TestCase):
    def test_model_smoke(self):
        test_model_smoke_and_logits_shape()

    def test_metrics(self):
        test_gzsl_metric_semantics()

    def test_backward(self):
        test_model_backward_smoke()

    def test_cuda_class_identity(self):
        test_model_accepts_cuda_class_identity()

    def test_identity(self):
        test_config_and_protocol_identity()

    def test_repository_boundary(self):
        test_repository_has_no_template_layer_or_legacy_experiment_tree()

    def test_git_binding(self):
        test_git_identity_is_bound_to_script_repository()

    def test_run_artifacts(self):
        test_output_boundary_and_epoch_artifact_semantics()

    def test_class_counts(self):
        test_standard_cub_class_counts()


if __name__ == "__main__":
    unittest.main()
