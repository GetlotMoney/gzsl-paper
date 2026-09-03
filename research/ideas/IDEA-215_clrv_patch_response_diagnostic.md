# IDEA-215：CLRV Patch Response Diagnostic

status: rejected_at_asset_signal_gate
problem_category: visual_grounding
problem: 当前V7全局CLS Reader与TG+GTD高度冗余；需要验证最终可追溯CLIP patch资产能否为候选对角色差异提供独立局部视觉证据。
mechanism_tags: [patch_response, candidate_pair, role_difference, diagnostic]

asset_identity:
- projected patch manifest: `/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/asset_manifest.json`
- manifest_sha256: `d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0`
- train_patch_sha256: `937a906d18cc7acc556e75fe8b9822e47be8cc6b3d21c89e181a80a257940537`
- current V7 dynamic CLS labels were rowwise equal to patch labels; CLS cosine mean/min: `0.9999978 / 0.9998829`.

minimal_falsification: On three deterministic seen-class folds, compare signed Top-2 patch evidence for fixed text-only Top-3 candidate pairs against CLS pair direction, image-level patch shuffle, and class-role-pair shuffle. A valid patch evidence signal must exceed shuffled controls; no official test image participates.

result:
- CLS pair-direction accuracy: `0.894623`
- true patch evidence: `1.000000`
- image-level patch shuffle: `0.999953`
- role-pair shuffle: `1.000000`

decision: The Top-2 patch score is saturated: any image/role relation can find a spuriously high local match. It contains no independent candidate-pair evidence, so CLRV raw patch retrieval is rejected. Do not continue Top-K patch reranking, raw patch attention, or score fusion from this path.

failure_boundary: This rejects the measured Top-2 high-dimensional patch similarity definition, not all possible externally processed visual interventions. Any future visual candidate must use a distinct evidence primitive and pass shuffled-control diagnostics before training.
