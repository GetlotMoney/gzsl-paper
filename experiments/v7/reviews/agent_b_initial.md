# Reviewer B initial review — FRAMEWORK-V7 promotion

- Reviewer: temporary Reviewer B
- Phase: stage-1 independent promotion review
- Frozen commit reviewed: `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`
- Promotion source: `2f7837266f4077b3fb7e40927fc6571499a76747`
- V6 RUN code commit: `8de7cebda0235ab12e1b4b8f669134c8f4e2c075`
- V6 reviewed training commit: `b707b0c4671051244cebf4f8404299fc016b281e`
- Config reviewed: `config/framework_v7.yaml`
- Declared V7 source checkpoint SHA256: `a551de9d182222141ab4be9db1ae2020417be3a7a7d1d4b369510d635f2207c9`
- Declared V6 metrics SHA256: `fbbd8ef520d8d6bca62cc1d860a0432a244ab99af30761a3ffd8c824f7c90879`
- Declared source training config SHA256: `73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`
- Declared tests: `32 passed`
- Constraint: I did not read any Agent A review file. I did not edit code or run a server. I created only this review file; the parent `experiments/v7/reviews/` directory did not exist and was created only to place this file at the requested path.

## Files and evidence read

- `AGENTS.md`
- `README.md`
- `model/frameworks/v7/__init__.py`
- `model/frameworks/v7/model.py`
- `model/frameworks/v7/evaluate.py`
- `config/framework_v7.yaml`
- `experiments/v7/FRAMEWORK.yaml`
- `experiments/v7/EXPERIMENT_QUEUE.csv`
- `experiments/v7/framework_diagram.html`
- `experiments/v7/tune/INDEX.md`
- `experiments/v7/ablation/INDEX.md`
- `experiments/v7/innovation/INDEX.md`
- `experiments/v7/confirmation/INDEX.md`
- `research/ideas/IDEA-201_compiled_pclr.md`
- `experiments/v6/EXPERIMENT_QUEUE.csv`
- `experiments/v6/innovation/INDEX.md`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/EXPERIMENT.yaml`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/result.md`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/MICRO_BATCH.md`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/CODE_REVIEW.md`
- Relevant V6/V7 promotion diff: `2f7837266f4077b3fb7e40927fc6571499a76747..568b01ec8a8e48ffe78336a6fc99f7708de03cbc`
- Relevant V7 deployment references in `model/frameworks/v6/compiled_pclr.py` and `model/frameworks/v6/train_compiled_pclr.py`

Read-only identity checks:

- `git rev-parse HEAD` returned `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`.
- `main` resolves to `52b511d77b4ad048f35b40dc3cbd9afd092167e9`.
- `main` is an ancestor of `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`, so a fast-forward of `main` is feasible.
- `framework/v7` does not resolve locally.
- `v7^{}` does not resolve locally.
- V6 tree file count under `experiments/v1..v6` was unchanged across `2f78372..568b01e` in my read-only count; V6 ledger files were modified, but no old experiment directories were deleted or renumbered.

## Summary verdict

`revise`.

The V7 deployment code, V7 config, V7 metrics, V6 provenance, paper-parent disclosure, official-test disclosure, and HTML/code computation story are broadly consistent. However, the formal Git promotion contract is not closed: the repository currently lacks both `framework/v7` and `v7`, while `README.md` and `experiments/v7/FRAMEWORK.yaml` already claim the frozen branch and tag exist. This is a hard promotion identity problem, not a model math problem.

## P0

### P0-1: `framework/v7` branch and `v7` tag are absent while the promotion files claim them as frozen refs

Evidence:

- `experiments/v7/FRAMEWORK.yaml` declares `framework_branch: framework/v7` and `framework_tag: v7`.
- `README.md` lists FRAMEWORK-V7 as having frozen branch `framework/v7` and Tag `v7`.
- Read-only Git checks show `framework/v7` does not resolve and `v7^{}` does not resolve.
- HEAD is `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`, currently on `codex/promote-compiled-pclr-v7`, not on a formal `framework/v7` ref.

Why this is P0:

The project rule says `framework/vX` and tag `vX` must fix the same formal framework commit and must not move. V7 is already documented as a formal promoted framework, so absent refs make the formal identity unverifiable and make README/FRAMEWORK provenance false at review time.

Closure condition:

- Create or verify `refs/heads/framework/v7` points exactly to `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`.
- Create or verify tag `v7` dereferences exactly to `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`.
- If branch/tag creation is intentionally scheduled after this review, change the current files to say `pending` instead of claiming frozen refs already exist, then freeze a new promotion commit and re-run the minimal Git-ref review.

## P1

None beyond the P0 Git identity blocker.

I did not find a P1 in the model/evaluation semantics:

- `model/frameworks/v7/model.py` loads only exported tensors and executes `h(x) @ Q.T + b`.
- V7 deployment does not import or call V6 training modules, online Top-K, relation edge scoring, or Laplacian solving.
- `model/frameworks/v7/evaluate.py` validates the source checkpoint SHA and the asset source config SHA, loads the V7 export through `load_v7_checkpoint()`, computes U/S/H in 200-class joint space, computes ZS by slicing unseen columns at the end, and checks exact promoted metrics.
- `load_v7_checkpoint()` verifies the checkpoint experiment id, source run code commit `8de7ceb...`, training config SHA `73a812...`, and `export` payload before building the standalone V7 model.
- `config/framework_v7.yaml`, `experiments/v7/FRAMEWORK.yaml`, and `experiments/v7/framework_diagram.html` consistently state the deployment shape `[200,1536]`, `bias [200]`, Reader `768-64-768`, no online Top-K, no relation-edge dependency, and no online Laplacian solve.
- The paper parent is explicitly TG+GTD, while V5 and matched online-V5 remain disclosed as repository development references, not erased from V6 result/provenance.
- Official-test selection and non-blind status are disclosed in config, framework metadata, README, and V6 source ledger.

## P2

### P2-1: V6 queue row now summarizes promotion but no longer carries the explicit program-drop phrase in the queue decision column

Evidence:

- In `experiments/v6/EXPERIMENT_QUEUE.csv`, V6-TRY-006 changed from `keep_owner_override` / `owner_override_keep_efficiency_candidate_program_gate_failed` to `promoted_framework_v7` / `owner_promote_svi_as_framework_v7_paper_parent_tg_gtd`.
- The detailed V6 `result.md` and `EXPERIMENT.yaml` still preserve `program_decision: drop_gate_b_contract_failed`, the negative deltas vs matched online-V5 and formal V5, and the owner override path.

Impact:

This does not alter the executable V7 deployment or erase the detailed V6 result, but the queue is the quick index readers will scan first. Without the explicit program-drop phrase there, the top-level queue can understate that V6 failed its original Gate-B accuracy contract.

Closure condition:

- In a later ledger-only update, append a compact phrase to the V6 queue decision such as `owner_promote_svi_as_framework_v7_paper_parent_tg_gtd_program_gate_failed_vs_v5_preserved`, or otherwise ensure the queue row links unambiguously to the preserved program drop. Do not change code identity for this alone.

### P2-2: V7 config does not explicitly record the promotion source commit `2f7837266f4077b3fb7e40927fc6571499a76747`

Evidence:

- `config/framework_v7.yaml` records `source_run_commit: 8de7ceb...`, `source_training_config_sha256: 73a812...`, checkpoint SHA, asset hashes, and owner decision.
- `experiments/v7/FRAMEWORK.yaml` records `reviewed_code_commit: b707b0c...`, `source_run_commit: 8de7ceb...`, checkpoint SHA, source metrics SHA, and repository source as `V6-TRY-006 / IDEA-201`.
- Neither file records the promotion source commit `2f7837266f4077b3fb7e40927fc6571499a76747` supplied for this audit.

Impact:

This is not a RUN/evaluation blocker because the executable artifact is keyed by checkpoint SHA and source run commit. It is a provenance gap for reproducing the exact ledger state from which the promotion commit was derived.

Closure condition:

- Add `promotion_source_commit: 2f7837266f4077b3fb7e40927fc6571499a76747` to `experiments/v7/FRAMEWORK.yaml` and, if desired, `config/framework_v7.yaml` in a ledger-only follow-up.

### P2-3: V7 evaluation does not verify `source_run_commit` from the YAML field directly

Evidence:

- `evaluate.py` validates source checkpoint SHA and asset config SHA from YAML.
- `load_v7_checkpoint()` independently verifies the checkpoint `code_commit` is `8de7ceb...` and config SHA is `73a812...`.
- The YAML field `source_run_commit` is present but not checked as a YAML field by `evaluate.py`.

Impact:

The effective identity is still protected by the checkpoint loader, so this is not P1. Directly checking the YAML field would make config drift fail earlier and produce a clearer error.

Closure condition:

- Add a YAML-level check that `config["source_run_commit"] == "8de7cebda0235ab12e1b4b8f669134c8f4e2c075"` and `config["source_training_config_sha256"] == "73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b"`.

### P2-4: General README still describes every RUN as producing `checkpoint_last.pth` and `data_fingerprints.json`, but V7 deployment evaluation is not a training RUN

Evidence:

- README's generic run section lists `training.log`, `metrics.json`, `model_best.pth`, `checkpoint_last.pth`, and `data_fingerprints.json` as minimum RUN artifacts.
- V7 evaluation is a standalone fixed-checkpoint evaluator, not a trainer; it prints metrics and verifies promoted values.

Impact:

This is a documentation generality issue, not a V7 blocker. It could confuse someone expecting the V7 evaluator to emit training artifacts.

Closure condition:

- Optionally clarify that the artifact list applies to training RUNs, while FRAMEWORK-V7 evaluation is a fixed deployed-checkpoint reproduction entry.

## Git and promotion contract review

- `main` fast-forward is feasible: `main` is currently `52b511d77b4ad048f35b40dc3cbd9afd092167e9`, and it is an ancestor of `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`.
- The required formal refs are not present: `framework/v7` and `v7` both fail to resolve.
- Because files already state `framework/v7` and `v7` as frozen refs, the review cannot pass until those refs exist and dereference to the frozen commit, or until the files are corrected to pending status.

## V6 preservation review

V6 was not deleted, moved, copied, or renumbered. The old experiment tree remains present. The V6 result still preserves:

- `program_decision: drop_gate_b_contract_failed`;
- owner override;
- promotion decision;
- negative deltas vs matched online-V5 and formal V5;
- source run commit, output URI, metrics SHA, evaluation history SHA, model SHA, and training log SHA.

The only concern is P2-1: the quick queue row now foregrounds promotion and no longer says program gate failed in the decision cell.

## Paper parent and internal provenance review

The paper parent is consistently changed to TG+GTD:

- `config/framework_v7.yaml` records `paper_parent.method: TG+GTD`, `run: TUNE-002-RUN-030`, `H: 79.070015`.
- `experiments/v7/FRAMEWORK.yaml` records `paper_parent_method: TG+GTD`, `paper_parent_run: TUNE-002-RUN-030`, and `paper_parent_H: 79.070015`.
- README states FRAMEWORK-V7 uses TG+GTD as the paper parent.

The internal V5/test-selection facts are not erased:

- README still lists FRAMEWORK-V5 and its formal V5 metrics.
- V7 metadata and limitations disclose nested official-test selection, non-blind status, and LLM relation text use.
- V6 result keeps the matched online-V5 and formal V5 comparison, including that C-PCLR was below both.

This satisfies the paper-parent/provenance disclosure requirement, subject to the P2 queue wording improvement.

## HTML and code consistency

The V7 HTML diagram is consistent with the code and config on the reviewed semantic points:

- Reader `768-64-768` matches `HIDDEN_DIM = 64`.
- `h=[norm(x),u]`, `Q=[200,1536]`, `b=[200]` matches `V7DeploymentModel.forward()`.
- No online Top-K, relation edge output, or Laplacian solve exists in V7 deployment code.
- Metrics table in HTML matches `config/framework_v7.yaml` and `experiments/v7/FRAMEWORK.yaml`.

The diagram line "Formal commit: pending promotion freeze" is acceptable only before branch/tag closure. If the files keep claiming frozen `framework/v7` and `v7`, the diagram should be updated after refs are created or changed to accurately state the frozen commit.

## Strongest counterexample

The strongest promotion-level counterexample is already present in the local repository state:

`README.md` and `experiments/v7/FRAMEWORK.yaml` claim `framework/v7` and `v7` as formal frozen refs, but the refs do not exist. A user checking out `framework/v7` or `v7` cannot reproduce the claimed formal framework identity. That breaks the "branch/tag same commit" contract even though the deployment code itself is coherent.

The strongest method-level counterexample remains disclosed rather than hidden: C-PCLR is below formal V5 and matched online-V5, so V7 can only be promoted under the owner's TG+GTD paper-parent and efficiency/architecture framing. It must not be claimed as higher accuracy than V5 until future evidence changes that.

## Final decision

Reviewer B stage-1 decision: `revise`.

- P0=1
- P1=0
- P2=4

The only blocking issue is Git formal identity. If `framework/v7` and `v7` are created and verified to dereference to `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`, I would downgrade P0-1 to closed without requiring code changes. After that, the remaining P2 items should not block promotion.
