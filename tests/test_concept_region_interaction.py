import numpy as np
import torch

from tools.diagnose_concept_region_interaction import (
    gaussian_blur,
    hierarchical_bootstrap,
    interaction_eta,
    magnitude_excess,
    perturb_windows,
    random_pair_like,
    select_nonoverlap_pair,
    window_edge_type,
    window_from_peak,
    windows_overlap,
)


def test_window_from_peak_uses_fixed_four_patch_window_at_edges():
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


def test_mean_fill_changes_only_fixed_56_pixel_window():
    image = torch.ones(3, 336, 336)
    blurred = gaussian_blur(image, 13, 4.0)
    result = perturb_windows(image, [(1, 2)], mode="mean_fill", blurred=blurred)
    mask = torch.zeros(336, 336, dtype=torch.bool)
    mask[14:70, 28:84] = True
    assert torch.all(result[:, mask] == 0)
    assert torch.all(result[:, ~mask] == 1)


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
