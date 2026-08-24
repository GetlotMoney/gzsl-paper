from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from model.paper_v2 import PaperV2ThreeModuleModel
from model.train_paper_v2 import random_batch_indices, stage_for_iteration
from tools.create_clip_asset_source_config import build_config as build_asset_source_config
from tools.gzsl_data import (
    clean_class_name,
    evaluate_prototypes,
    load_xlsa_split,
    resolve_xlsa_image_path,
)


class PaperV2ModelTest(unittest.TestCase):
    def make_model(self, class_count=50, seen_count=37, **modes):
        generator = torch.Generator().manual_seed(901 + class_count)
        sentences = torch.randn(class_count, 8, 768, generator=generator)
        centroids = F.normalize(
            torch.randn(seen_count, 768, generator=generator), dim=-1
        )
        return PaperV2ThreeModuleModel(
            sentences,
            torch.arange(seen_count),
            centroids,
            dropout=0.0,
            **modes,
        )

    def test_dynamic_awa2_and_sun_class_axes(self):
        awa2 = self.make_model(50, 37)
        sun = self.make_model(717, 645)
        self.assertEqual(tuple(awa2.prototypes().shape), (50, 768))
        self.assertEqual(tuple(sun.prototypes().shape), (717, 768))

    def test_module_off_paths_are_exact(self):
        model = self.make_model(
            tg_vpr_mode="full",
            transport_mode="off",
            ccgr_mode="off",
        ).eval()
        stages = model.prototype_stages()
        self.assertTrue(torch.equal(stages["tg_vpr"], stages["transported"]))
        self.assertTrue(torch.equal(stages["transported"], stages["final"]))
        self.assertEqual(float(stages["transport_step"].abs().max()), 0.0)
        self.assertEqual(float(stages["generator_magnitude"].abs().max()), 0.0)

    def test_ccgr_initially_returns_transport_parent(self):
        model = self.make_model(
            transport_mode="tangent_ntr",
            ccgr_mode="class_conditioned_four",
        ).eval()
        stages = model.prototype_stages()
        self.assertTrue(
            torch.allclose(stages["transported"], stages["final"], atol=1e-6, rtol=1e-6)
        )
        self.assertEqual(float(stages["transport_step"].abs().max().detach()), 0.0)
        self.assertEqual(float(stages["generator_magnitude"].abs().max().detach()), 0.0)

    def test_internal_ablation_modes_have_expected_parameter_groups(self):
        shared = self.make_model(ccgr_mode="shared")
        groups = shared.parameter_groups()
        self.assertTrue(groups["ccgr_shared"])
        self.assertFalse(groups["ccgr_class"])
        tangent = self.make_model(transport_mode="tangent", ccgr_mode="off")
        groups = tangent.parameter_groups()
        self.assertTrue(groups["transport"])
        self.assertFalse(groups["ntr"])
        grouped = self.make_model(
            tg_vpr_mode="grouped_no_value", transport_mode="off", ccgr_mode="off"
        )
        self.assertFalse(grouped.parameter_groups()["tg_vpr"])

    def test_stage_boundaries_scale_with_dataset_size(self):
        ntrain = 37
        self.assertEqual(stage_for_iteration(ntrain, 0)[0], "TG_ONLY")
        self.assertEqual(stage_for_iteration(ntrain, 36)[0], "TG_ONLY")
        self.assertEqual(stage_for_iteration(ntrain, 37)[0], "TRANSFER_CCGR")
        self.assertEqual(stage_for_iteration(ntrain, 110)[0], "TRANSFER_CCGR")
        self.assertEqual(stage_for_iteration(ntrain, 111)[0], "JOINT_FINETUNE")
        self.assertEqual(stage_for_iteration(ntrain, 147)[0], "JOINT_FINETUNE")

    def test_cached_topology_reference_matches_direct_formula(self):
        model = self.make_model(50, 37).eval()
        adapted = model.prototypes()
        base = model.tg_vpr.base_prototypes()
        off_diag = ~torch.eye(50, dtype=torch.bool)
        x = (base @ base.T).detach()[off_diag]
        y = (adapted @ adapted.T)[off_diag]
        x = x - x.mean()
        y = y - y.mean()
        direct = 1.0 - (x * y).sum() / (
            torch.sqrt(x.square().sum() + 1e-8)
            * torch.sqrt(y.square().sum() + 1e-8)
        )
        self.assertTrue(torch.allclose(model.topology_loss(adapted), direct, atol=1e-7, rtol=1e-7))

    def test_chen_random_batch_contract_is_dataset_agnostic(self):
        generator = torch.Generator().manual_seed(7)
        first = random_batch_indices(1234, 50, generator)
        second = random_batch_indices(1234, 50, generator)
        self.assertEqual(first.unique().numel(), 50)
        self.assertFalse(torch.equal(first, second))


class PaperV2DataTest(unittest.TestCase):
    def test_asset_source_config_binds_each_dataset_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            role_texts = Path(temporary) / "roles.json"
            role_texts.write_text("{}", encoding="utf-8")
            awa2 = build_asset_source_config("AWA2", role_texts)
            sun = build_asset_source_config("SUN", role_texts)
            self.assertTrue(awa2["raw_archive"].endswith("AwA2-data.zip"))
            self.assertTrue(sun["raw_archive"].endswith("SUNAttributeDB_Images.tar.gz"))
            self.assertNotEqual(
                awa2["expected_sha256"]["raw_archive"],
                sun["expected_sha256"]["raw_archive"],
            )

    def test_double_slash_xlsa_path_resolves_against_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "Animals_with_Attributes2" / "JPEGImages" / "antelope" / "a.jpg"
            expected.parent.mkdir(parents=True)
            expected.touch()
            resolved = resolve_xlsa_image_path(
                root,
                "/machine/data/Animals_with_Attributes2//JPEGImages/antelope/a.jpg",
                ["Animals_with_Attributes2/JPEGImages"],
            )
            self.assertEqual(resolved, expected)

    def test_generic_xlsa_split_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_files = np.empty((6, 1), dtype=object)
            for index in range(6):
                image_files[index, 0] = f"/dataset/images/image_{index}.jpg"
            sio.savemat(
                root / "res101.mat",
                {
                    "labels": np.array([[1], [1], [2], [2], [3], [3]]),
                    "image_files": image_files,
                },
            )
            class_names = np.empty((3, 1), dtype=object)
            class_names[:, 0] = ["001.class_one", "class_two", "class_three"]
            sio.savemat(
                root / "att_splits.mat",
                {
                    "trainval_loc": np.array([[1], [3]]),
                    "test_seen_loc": np.array([[2], [4]]),
                    "test_unseen_loc": np.array([[5], [6]]),
                    "allclasses_names": class_names,
                },
            )
            split = load_xlsa_split(root / "res101.mat", root / "att_splits.mat")
            self.assertEqual(split.class_count, 3)
            self.assertTrue(torch.equal(split.seen_classes, torch.tensor([0, 1])))
            self.assertTrue(torch.equal(split.unseen_classes, torch.tensor([2])))
            self.assertEqual(clean_class_name(split.class_names[0]), "class one")

            prototypes = torch.eye(3, 768)
            seen_features = prototypes[:2]
            unseen_features = prototypes[2:]
            metrics = evaluate_prototypes(
                prototypes,
                1.0,
                seen_features,
                torch.tensor([0, 1]),
                unseen_features,
                torch.tensor([2]),
                torch.tensor([0, 1]),
                torch.tensor([2]),
                device=torch.device("cpu"),
            )
            self.assertEqual(metrics, {"U": 100.0, "S": 100.0, "H": 100.0, "ZS": 100.0})


if __name__ == "__main__":
    unittest.main()
