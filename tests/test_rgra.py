from __future__ import annotations

import torch
import torch.nn.functional as F

from model.frameworks.v6.rgra import (
    PATCH_COUNT,
    RGRA_CONDITIONS,
    RGRAModel,
    build_relation_field,
    mean8_prototypes,
    raw_role_queries,
)


def _edges() -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for class_id in range(150):
        pair = tuple(sorted((class_id, (class_id + 1) % 150)))
        if pair not in used:
            used.add(pair)
            pairs.append(pair)
    for a_id in range(200):
        for b_id in range(a_id + 1, 200):
            pair = (a_id, b_id)
            if pair not in used:
                used.add(pair)
                pairs.append(pair)
            if len(pairs) == 438:
                return torch.tensor(pairs, dtype=torch.long)
    raise AssertionError("edge fixture incomplete")


def _fixture() -> tuple[RGRAModel, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(205)
    sentences = F.normalize(torch.randn(200, 8, 768, generator=generator), dim=-1)
    v5 = F.normalize(torch.randn(200, 768, generator=generator), dim=-1)
    relations = F.normalize(torch.randn(438, 2, 768, generator=generator), dim=-1)
    model = RGRAModel(sentences, v5, relations, _edges(), torch.arange(150))
    cls = torch.randn(4, 768, generator=generator)
    patches = torch.randn(4, 36, 768, generator=generator)
    targets = torch.tensor([0, 1, 2, 149])
    return model, cls, patches, targets


def test_raw_6_plus_1_plus_1_text_contract_and_s_off_exact_path():
    generator = torch.Generator().manual_seed(1)
    sentences = F.normalize(torch.randn(200, 8, 768, generator=generator), dim=-1)
    queries = raw_role_queries(sentences)
    assert tuple(queries.shape) == (200, 3, 768)
    assert torch.allclose(queries.norm(dim=-1), torch.ones(200, 3), atol=1e-6)
    assert torch.allclose(
        queries[:, 0], F.normalize(sentences[:, :6].mean(dim=1), dim=-1), atol=1e-7
    )
    assert torch.allclose(queries[:, 1], F.normalize(sentences[:, 6], dim=-1), atol=1e-7)
    assert torch.allclose(queries[:, 2], F.normalize(sentences[:, 7], dim=-1), atol=1e-7)

    model, _, _, _ = _fixture()
    assert torch.equal(model.rsc.prototypes(s_off=True), model.rsc.p_mean8)
    assert torch.equal(model.rsc.role_queries(s_off=True), model.rsc.q_raw)
    assert torch.equal(model.rsc.group_weights(s_off=True), torch.full((3,), 1.0 / 3.0))


def test_relation_field_is_compiled_and_raw_graph_assets_are_not_persisted():
    generator = torch.Generator().manual_seed(2)
    relations = F.normalize(torch.randn(438, 2, 768, generator=generator), dim=-1)
    edges = _edges()
    field = build_relation_field(relations, edges)
    assert tuple(field.shape) == (200, 768)
    assert torch.isfinite(field).all()
    assert torch.allclose(field.norm(dim=-1), torch.ones(200), atol=1e-5)

    model, _, _, _ = _fixture()
    state_keys = set(model.state_dict())
    assert "rfm.relation_field" in state_keys
    assert not {"relation_embeddings", "edge_index", "incidence", "laplacian_map"} & state_keys
    package = model.export_graph_free_state()
    export_keys = set(package["state_dict"])
    assert "rfm.relation_field" in export_keys
    assert not {"relation_embeddings", "edge_index", "incidence", "laplacian_map"} & export_keys


def test_graph_free_export_roundtrip_matches_full_logits():
    model, cls, patches, _ = _fixture()
    model.eval()
    package = model.export_graph_free_state()
    deployed = RGRAModel.from_graph_free_state(package).eval()
    with torch.no_grad():
        expected = model.logits(cls, patches, condition="full")
        actual = deployed.logits(cls, patches, condition="full")
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_v5_reader_initialization_is_loaded_into_rfm():
    model, _, _, _ = _fixture()
    state = {
        "reader_in.weight": torch.full_like(model.rfm.reader.in_proj.weight, 0.125),
        "reader_in.bias": torch.full_like(model.rfm.reader.in_proj.bias, -0.25),
        "reader_out.weight": torch.full_like(model.rfm.reader.out_proj.weight, 0.375),
        "reader_out.bias": torch.full_like(model.rfm.reader.out_proj.bias, -0.5),
    }
    replacement = RGRAModel(
        model.rsc.role_sentence_embeds,
        model.rsc.p_v5,
        model.relation_embeddings,
        model.edge_index,
        model.seen_classes,
        class_count=model.class_count,
        reader_state_dict=state,
    )
    assert torch.equal(replacement.rfm.reader.in_proj.weight, state["reader_in.weight"])
    assert torch.equal(replacement.rfm.reader.in_proj.bias, state["reader_in.bias"])
    assert torch.equal(replacement.rfm.reader.out_proj.weight, state["reader_out.weight"])
    assert torch.equal(replacement.rfm.reader.out_proj.bias, state["reader_out.bias"])


def test_full_and_all_off_control_paths_have_fixed_shapes_and_attention():
    model, cls, patches, _ = _fixture()
    for condition in sorted(RGRA_CONDITIONS):
        output = model.score_components(cls, patches, condition=condition)
        assert tuple(output["logits"].shape) == (4, 200)
        assert tuple(output["semantic_logits"].shape) == (4, 200)
        assert tuple(output["visual_logits"].shape) == (4, 200)
        assert tuple(output["interaction_logits"].shape) == (4, 200)
        assert tuple(output["attention"].shape) == (4, 200, 3, PATCH_COUNT)
        assert torch.isfinite(output["logits"]).all()
        assert torch.allclose(
            output["attention"].sum(dim=-1),
            torch.ones(4, 200, 3),
            atol=1e-6,
        )

    v_off = model.score_components(cls, patches, condition="v_off")
    assert torch.equal(v_off["visual_logits"], torch.zeros_like(v_off["visual_logits"]))
    assert torch.equal(v_off["support_gate"], torch.full_like(v_off["support_gate"], 0.5))
    assert not torch.equal(v_off["relation_scores"], torch.zeros_like(v_off["relation_scores"]))

    i_off = model.score_components(cls, patches, condition="i_off")
    assert torch.equal(i_off["interaction_logits"], torch.zeros_like(i_off["interaction_logits"]))


def test_alpha_zero_full_is_exact_i_off_and_class_slice_is_late():
    model, cls, patches, _ = _fixture()
    alpha_zero = model.logits(cls, patches, condition="full", alpha_override=0.0)
    i_off = model.logits(cls, patches, condition="i_off")
    assert torch.equal(alpha_zero, i_off)

    ids = torch.tensor([7, 2, 180, 151])
    full = model.logits(cls, patches)
    subset = model.logits(cls, patches, ids)
    assert torch.equal(subset, full.index_select(1, ids))


def test_additive_and_shuffled_controls_preserve_formula_but_change_grounding():
    model, cls, patches, _ = _fixture()
    full = model.score_components(cls, patches, condition="full")
    additive = model.score_components(cls, patches, condition="additive")
    shuffled = model.score_components(cls, patches, condition="shuffled")

    assert torch.equal(additive["support_for_relation"], torch.full_like(additive["support_for_relation"], 0.5))
    assert torch.equal(
        additive["interaction_logits"],
        model.rfm.alpha() * 0.5 * additive["relation_scores"],
    )
    assert torch.equal(shuffled["support_for_relation"].sort(dim=1).values, full["support_gate"].sort(dim=1).values)
    assert not torch.equal(shuffled["support_for_relation"], full["support_gate"])


def test_cls_only_loss_sends_nonzero_gradients_to_all_three_deployed_modules():
    model, cls, patches, targets = _fixture()
    norms = model.classification_gradient_norms(cls, patches, targets)
    assert norms["classification_loss"] > 0.0
    assert norms["rsc"] > 0.0
    assert norms["rva"] > 0.0
    assert norms["rfm"] > 0.0


def test_total_loss_and_input_guards_cover_training_contract():
    model, cls, patches, targets = _fixture()
    total, parts = model.total_loss(cls, patches, targets)
    assert torch.isfinite(total)
    assert set(parts) == {"classification_loss", "topology_loss", "direction_loss"}
    assert all(torch.isfinite(value) for value in parts.values())

    bad_patches = torch.randn(4, 35, 768)
    try:
        model.logits(cls, bad_patches)
    except ValueError as exc:
        assert "36" in str(exc)
    else:
        raise AssertionError("RGRA must reject non-36-patch inputs.")

    try:
        model.classification_loss(cls, patches, torch.tensor([0, 1, 2, 199]))
    except ValueError as exc:
        assert "seen-class" in str(exc)
    else:
        raise AssertionError("RGRA must reject true-unseen labels in CE.")
