import json
import os

import numpy as np
import pytest
import torch

from tools.diagnose_content_aware_inpainted_interaction import (
    CUBLAS_WORKSPACE_CONFIG,
    boundary_reflect_inpaint,
    choose_hard_control,
    comparable_environment,
    fixed_attention_evidence,
    harmonic_inpaint,
    hierarchical_bootstrap,
    interaction_eta,
    magnitude_excess,
    pair_variants,
    random_pair_like,
    select_nonoverlap_pair,
    validate_assets,
    window_pixel_mask,
    window_edge_type,
    window_from_peak,
    windows_overlap,
)
from tools.runtime import sha256_file


def test_window_from_peak_uses_fixed_four_patch_window_at_edges():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert window_from_peak(0) == (0, 0)
    assert window_from_peak(23) == (0, 20)
    assert window_from_peak(23 * 24) == (20, 0)
    assert window_from_peak(24 * 24 - 1) == (20, 20)
    assert window_from_peak(12 * 24 + 12) == (11, 11)


def test_nonoverlap_pair_skips_adjacent_peaks_that_map_to_overlapping_windows():
    attention = np.zeros(576, dtype=np.float64)
    attention[12 * 24 + 12] = 0.9
    attention[12 * 24 + 13] = 0.8
    attention[2 * 24 + 2] = 0.7
    pair = select_nonoverlap_pair(attention)
    assert pair is not None
    assert pair["windows"][0] == (11, 11)
    assert pair["windows"][1] == (1, 1)
    assert not windows_overlap(*pair["windows"])


def test_eta_sign_matches_complement_and_redundancy_examples():
    complement = interaction_eta(10.0, 2.0, 2.0, 0.0)
    redundancy = interaction_eta(10.0, 9.0, 9.0, 0.0)
    assert complement == -6.0
    assert redundancy == 8.0
    assert magnitude_excess(complement, 1.0) == 5.0
    assert magnitude_excess(redundancy, -2.0) == 6.0


def test_fixed_original_attention_keeps_independent_patch_effects_additive():
    original_similarity = torch.tensor([[1.0, 0.9, 0.0]])
    original_attention = torch.tensor([0.5, 0.5, 0.0])
    original = fixed_attention_evidence(original_similarity, original_attention).item()
    score_a = fixed_attention_evidence(torch.tensor([[0.0, 0.9, 0.0]]), original_attention).item()
    score_b = fixed_attention_evidence(torch.tensor([[1.0, 0.0, 0.0]]), original_attention).item()
    score_union = fixed_attention_evidence(torch.zeros(1, 3), original_attention).item()
    assert abs(interaction_eta(original, score_a, score_b, score_union)) < 1e-7


def test_hard_control_rejects_frequency_mismatch_even_with_identical_attention():
    attention = np.full(576, 1e-6, dtype=np.float64)
    attention[2 * 24 + 2] = 0.4
    attention[12 * 24 + 12] = 0.3
    attentions = np.stack([attention, attention])
    target_pair = select_nonoverlap_pair(attention)
    clusters = [(0, [1, 2]), (0, list(range(50, 150)))]
    result = choose_hard_control(
        target_pair=target_pair,
        target_concept=0,
        class_id=1,
        attentions=attentions,
        clusters=clusters,
        grid_side=24,
        window_side=4,
        maximum_log_distance=0.75,
        maximum_frequency_log_distance=np.log(2.0),
    )
    assert result is None


def test_random_control_matches_edge_types_and_is_nonoverlapping():
    target = {"windows": [(0, 5), (10, 10)]}
    random_pair = random_pair_like(target, grid_side=24, window_side=4, seed=7)
    assert random_pair is not None
    target_types = sorted(window_edge_type(window) for window in target["windows"])
    random_types = sorted(window_edge_type(window) for window in random_pair["windows"])
    assert random_types == target_types
    assert not windows_overlap(*random_pair["windows"])
    assert not set(random_pair["windows"]) & set(target["windows"])
    assert all(
        not windows_overlap(random_window, target_window)
        for random_window in random_pair["windows"]
        for target_window in target["windows"]
    )


def test_content_aware_inpainting_replaces_only_mask_from_outside_context():
    image = torch.zeros(3, 336, 336)
    windows = [(1, 2), (10, 12)]
    mask = window_pixel_mask(image, windows)
    image[:, mask] = 999.0
    harmonic = harmonic_inpaint(image, windows, iterations=64)
    reflected = boundary_reflect_inpaint(image, windows)
    for result in (harmonic, reflected):
        assert torch.equal(result[:, ~mask], image[:, ~mask])
        assert torch.isfinite(result).all()
        assert float(result[:, mask].abs().max()) < 1.0


def test_boundary_reflect_handles_image_edge_without_reusing_masked_content():
    image = torch.zeros(3, 336, 336)
    windows = [(0, 0)]
    mask = window_pixel_mask(image, windows)
    image[:, mask] = 999.0
    result = boundary_reflect_inpaint(image, windows)
    assert float(result[:, mask].abs().max()) == 0.0


@pytest.mark.parametrize("mode", ["harmonic_inpaint_64", "boundary_reflect_inpaint"])
@pytest.mark.parametrize("windows", [[(1, 1), (1, 5)], [(1, 1), (14, 16)]])
def test_pair_variants_reuse_one_replacement_for_exact_composition(mode, windows):
    image = torch.arange(3 * 336 * 336, dtype=torch.float32).reshape(3, 336, 336) / 1000.0
    pair = {"windows": windows}
    variant_a, variant_b, variant_union = pair_variants(image, pair, mode, 64, 4)
    mask_a = window_pixel_mask(image, [windows[0]])
    mask_b = window_pixel_mask(image, [windows[1]])
    expected = image.clone()
    expected[:, mask_a] = variant_a[:, mask_a]
    expected[:, mask_b] = variant_b[:, mask_b]
    assert torch.equal(variant_union, expected)
    assert torch.equal(variant_union[:, mask_a], variant_a[:, mask_a])
    assert torch.equal(variant_union[:, mask_b], variant_b[:, mask_b])


def test_hierarchical_bootstrap_is_deterministic_and_class_balanced():
    values = np.asarray([1.0, 1.0, 3.0, 3.0])
    classes = np.asarray([0, 0, 1, 1])
    images = np.asarray([10, 11, 20, 21])
    first = hierarchical_bootstrap(values, classes, images, replicates=200, seed=7)
    second = hierarchical_bootstrap(values, classes, images, replicates=200, seed=7)
    assert first == second
    assert first["point"] == 2.0
    assert first["class_count"] == 2
    assert first["image_count"] == 4
    reordered = hierarchical_bootstrap(
        values[::-1], classes[::-1], images[::-1], replicates=200, seed=7
    )
    assert reordered == first


def test_asset_validation_checks_actual_checkpoint_and_cache_bytes(tmp_path):
    role = tmp_path / "role.json"
    source = tmp_path / "source.yaml"
    checkpoint = tmp_path / "clip.pt"
    labels = tmp_path / "train_labels.pt"
    patches = tmp_path / "train_patch_features.npy"
    parent_raw = tmp_path / "parent-raw.json"
    parent_contrastive = tmp_path / "parent-contrastive.json"
    role.write_bytes(b"role")
    source.write_bytes(b"source")
    checkpoint.write_bytes(b"clip")
    labels.write_bytes(b"labels")
    patches.write_bytes(b"patches")
    parent_raw.write_bytes(b"raw")
    parent_contrastive.write_bytes(b"contrastive")
    manifest = tmp_path / "asset_manifest.json"
    payload = {
        "schema_version": "gzsl-paper.projected-patch-assets.v1",
        "patch_shape": [576, 768],
        "clip_checkpoint_sha256": sha256_file(checkpoint),
        "outputs_sha256": {
            "train_labels.pt": sha256_file(labels),
            "train_patch_features.npy": sha256_file(patches),
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "visual_asset_manifest": str(manifest),
        "visual_asset_manifest_sha256": sha256_file(manifest),
        "role_texts": str(role),
        "role_texts_sha256": sha256_file(role),
        "source_config": str(source),
        "source_config_sha256": sha256_file(source),
        "clip_checkpoint": str(checkpoint),
        "clip_checkpoint_sha256": "0" * 64,
        "train_labels": str(labels),
        "final_patches": str(patches),
        "parent_raw_result_uri": str(parent_raw),
        "parent_raw_result_sha256": sha256_file(parent_raw),
        "parent_contrastive_result_uri": str(parent_contrastive),
        "parent_contrastive_result_sha256": sha256_file(parent_contrastive),
    }
    with pytest.raises(ValueError, match="checkpoint SHA"):
        validate_assets(config)
    config["clip_checkpoint_sha256"] = sha256_file(checkpoint)
    patches.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="内容SHA"):
        validate_assets(config)


def test_environment_comparison_ignores_only_gpu_uuid():
    left = {"torch": "2", "gpu_name": "4090", "gpu_uuid": "A"}
    right = {"torch": "2", "gpu_name": "4090", "gpu_uuid": "B"}
    wrong = {"torch": "3", "gpu_name": "4090", "gpu_uuid": "B"}
    assert comparable_environment(left) == comparable_environment(right)
    assert comparable_environment(left) != comparable_environment(wrong)
