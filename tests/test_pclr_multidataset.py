from pathlib import Path

from model.innovations.evaluate_pclr_multidataset import load_multidataset_config


def test_awa2_and_sun_generic_pclr_configs_are_fixed():
    expected = {
        "AWA2": ("config/tries/v4_confirm_003_pclr_awa2.yaml", 5, 0.05),
        "SUN": ("config/tries/v4_confirm_003_pclr_sun.yaml", 60, 0.15),
    }
    for dataset, (path, candidate_top_k, gamma) in expected.items():
        config, digest = load_multidataset_config(Path(path))
        assert config["dataset"] == dataset
        assert config["candidate_top_k"] == candidate_top_k
        assert config["seen_logit_gamma"] == gamma
        assert config["nested_official_test_selection"] is True
        assert config["generic_class_name_directions"] is True
        assert config["llm_world_knowledge_used"] is False
        assert len(digest) == 64
