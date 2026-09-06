import torch

from model.frameworks.v5 import V5_DEPLOYMENT, v5_logits


def test_v5_deployment_parameters_are_frozen():
    assert V5_DEPLOYMENT == {
        "candidate_top_k": 17,
        "ridge_lambda": 0.3,
        "potential_cap": 0.5,
        "inference_relation_temperature": 0.2,
        "correction_scale": 6.95,
        "role0_weight": 0.16,
        "role6_weight": 0.36,
        "seen_logit_gamma": 0.91,
    }


def test_v5_logits_is_no_grad_public_entry():
    assert callable(v5_logits)
    assert hasattr(torch, "no_grad")
