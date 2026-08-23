from pathlib import Path
import unittest

import torch
import torch.nn as nn

from model.innovations.train_jscf import enable_joint_parameters, load_config


ROOT = Path(__file__).resolve().parents[1]


class DummySDRS(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.zeros(()))
        self.unused = nn.Parameter(torch.ones(()))


class DummyCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_gamma = nn.Parameter(torch.zeros(()))


class DummySDCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weight_residual = nn.Parameter(torch.zeros(8))
        self.unused = nn.Parameter(torch.ones(()))


class JSCFTest(unittest.TestCase):
    def test_only_ten_registered_parameters_are_trainable(self):
        sdrs, calibrator, sdcr = DummySDRS(), DummyCalibrator(), DummySDCR()
        names = enable_joint_parameters(sdrs, calibrator, sdcr)
        self.assertEqual(
            names,
            ["sdrs.raw_slope", "calibrator.raw_gamma", "sdcr.raw_weight_residual"],
        )
        trainable = sum(
            parameter.numel()
            for module in (sdrs, calibrator, sdcr)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, 10)
        self.assertFalse(sdrs.unused.requires_grad)
        self.assertFalse(sdcr.unused.requires_grad)

    def test_config_binds_joint_parent_and_chen_boundary(self):
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-050_jscf/configs/RUN-001.yaml"
        )
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["sdcr_kl_weight"], 0.01)
        self.assertEqual(config["niters"], 28228)
        self.assertFalse(config["unseen_images_used_for_gradient"])

    def test_rescue_freezes_sebc_and_keeps_nine_parameters(self):
        sdrs, calibrator, sdcr = DummySDRS(), DummyCalibrator(), DummySDCR()
        names = enable_joint_parameters(sdrs, calibrator, sdcr, train_sebc=False)
        self.assertEqual(names, ["sdrs.raw_slope", "sdcr.raw_weight_residual"])
        self.assertFalse(calibrator.raw_gamma.requires_grad)
        trainable = sum(
            parameter.numel()
            for module in (sdrs, calibrator, sdcr)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, 9)
        config, _ = load_config(
            ROOT / "experiments/v2/innovation/INNOVATION-050_jscf/configs/RUN-002.yaml"
        )
        self.assertFalse(config["train_sebc"])


if __name__ == "__main__":
    unittest.main()
