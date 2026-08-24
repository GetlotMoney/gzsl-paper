from pathlib import Path
import unittest

import torch

from model.innovations.train_threefold_validation import (
    aggregate_fold_histories,
    build_fold_splits,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ThreefoldValidationTest(unittest.TestCase):
    def test_fold_split_keeps_pseudo_unseen_out_of_fit(self):
        labels = torch.arange(6).repeat_interleave(10)
        classes = torch.arange(6)
        folds = []
        for start in (0, 2, 4):
            pseudo_unseen = classes[start : start + 2]
            pseudo_seen = classes[~torch.isin(classes, pseudo_unseen)]
            folds.append((pseudo_seen, pseudo_unseen))
        splits = build_fold_splits(labels, folds, seed=5, holdout_fraction=0.2)
        self.assertEqual(len(splits), 3)
        for split in splits:
            fit_labels = labels.index_select(0, split["fit_positions"])
            self.assertFalse(torch.isin(fit_labels, split["pseudo_unseen"]).any())
            self.assertFalse(
                torch.isin(split["fit_positions"], split["val_seen_positions"]).any()
            )
            self.assertFalse(
                torch.isin(split["fit_positions"], split["val_unseen_positions"]).any()
            )

    def test_aggregate_selects_epoch_by_mean_h_not_single_fold_max(self):
        histories = []
        values = ((70.0, 80.0), (75.0, 78.0), (80.0, 76.0))
        for first, second in values:
            histories.append(
                [
                    {"validation_metrics_percent": {"U": first, "S": first, "H": first, "ZS": first}},
                    {"validation_metrics_percent": {"U": second, "S": second, "H": second, "ZS": second}},
                ]
            )
        aggregate, selected = aggregate_fold_histories(histories)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(selected["epoch"], 2)
        self.assertAlmostEqual(selected["mean_metrics_percent"]["H"], 78.0)
        self.assertEqual(selected["range_H"], 4.0)

    def test_config_excludes_official_inputs_and_pseudo_unseen_gradients(self):
        config, _ = load_config(
            ROOT
            / "experiments/v2/tune/TUNE-003_pure_threefold_hparams/configs/RUN-001.yaml"
        )
        self.assertFalse(config["official_test_loaded"])
        self.assertFalse(config["test_used_for_selection"])
        self.assertFalse(config["pseudo_unseen_images_used_for_gradient"])
        self.assertNotIn("seen_features", config["inputs"])
        self.assertNotIn("unseen_features", config["inputs"])
        topology_low, _ = load_config(
            ROOT
            / "experiments/v2/tune/TUNE-003_pure_threefold_hparams/configs/RUN-002.yaml"
        )
        topology_high, _ = load_config(
            ROOT
            / "experiments/v2/tune/TUNE-003_pure_threefold_hparams/configs/RUN-003.yaml"
        )
        self.assertEqual(topology_low["topology_weight"], 0.03)
        self.assertEqual(topology_high["topology_weight"], 0.2)

    def test_source_does_not_reference_official_test_cache(self):
        source = (
            ROOT / "model/innovations/train_threefold_validation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("CUB_test_seen", source)
        self.assertNotIn("CUB_test_unseen", source)
        self.assertNotIn("test_seen_loc", source)
        self.assertNotIn("test_unseen_loc", source)


if __name__ == "__main__":
    unittest.main()
