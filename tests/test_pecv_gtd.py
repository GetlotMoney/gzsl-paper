from pathlib import Path

import torch
import yaml

from model.innovations.pecv_gtd import (
    PairwiseErrorCorrectingVerifier,
    corrected_topk_scores,
    stable_topk_ids,
)
from model.innovations.train_pecv_gtd import ThreeGroupSchedule, load_config


def test_zero_initialization_is_exact_module_off():
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(3, 12, generator=generator)
    prototypes = torch.randn(6, 12, generator=generator)
    roles = torch.randn(6, 8, 12, generator=generator)
    candidates = torch.tensor([[0, 1, 2, 3, 4]]).repeat(3, 1)
    parent = torch.randn(3, 5, generator=generator)
    verifier = PairwiseErrorCorrectingVerifier()
    full = corrected_topk_scores(parent, images, candidates, prototypes, roles, verifier)
    torch.testing.assert_close(full, parent, rtol=0, atol=0)
    assert corrected_topk_scores(parent, images, candidates, prototypes, roles, None) is parent


def test_antisymmetry_and_zero_sum_after_nonzero_weights():
    generator = torch.Generator().manual_seed(8)
    images = torch.randn(2, 12, generator=generator)
    prototypes = torch.randn(5, 12, generator=generator)
    roles = torch.randn(5, 8, 12, generator=generator)
    candidates = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    parent = torch.randn(2, 5, generator=generator)
    verifier = PairwiseErrorCorrectingVerifier()
    with torch.no_grad():
        verifier.reader[-1].weight.fill_(0.1)
    ab = verifier.correction(images, prototypes[[0, 1]], prototypes[[1, 0]], roles[[0, 1]], roles[[1, 0]])
    ba = verifier.correction(images, prototypes[[1, 0]], prototypes[[0, 1]], roles[[1, 0]], roles[[0, 1]])
    torch.testing.assert_close(ab, -ba, rtol=0, atol=1e-6)
    full = corrected_topk_scores(parent, images, candidates, prototypes, roles, verifier)
    torch.testing.assert_close((full - parent).sum(1), torch.zeros(2), rtol=0, atol=1e-6)


def test_stable_topk_uses_global_class_id_tie_break():
    logits = torch.tensor([[1.0, 1.0, 0.0]])
    ids = torch.tensor([20, 10, 30])
    assert stable_topk_ids(logits, ids, 3).tolist() == [[10, 20, 30]]


def test_formal_configs_are_fixed200_and_from_scratch():
    root = Path(__file__).resolve().parents[1]
    for name, weight in (
        ("v4_try_020_pecv_full_fixed200.yaml", 1.0),
        ("v4_try_020_pecv_parent_fixed200.yaml", 0.0),
    ):
        config, _ = load_config(root / "config" / "tries" / name)
        assert config["nominal_epochs"] == 200
        assert config["total_updates"] == 28228
        assert config["pecv_loss_weight"] == weight
        assert config["tg_checkpoint"] is None
        assert config["gtd_checkpoint"] is None
        assert config["pecv_checkpoint"] is None


def test_three_group_schedule_preserves_tg_and_warms_new_modules():
    parameters = [torch.nn.Parameter(torch.zeros(())) for _ in range(3)]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters[0]], "lr": 1e-4},
            {"params": [parameters[1]], "lr": 1e-4},
            {"params": [parameters[2]], "lr": 1e-3},
        ]
    )
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/tries/v4_try_020_pecv_full_fixed200.yaml").read_text()
    )
    schedule = ThreeGroupSchedule(optimizer, config)
    assert schedule.multipliers(1) == [1.0, 0.1, 0.1]
    assert schedule.multipliers(schedule.warmup_updates) == [1.0, 1.0, 1.0]
    final = schedule.multipliers(28228)
    assert final == [1.0, 0.1, 0.1]
