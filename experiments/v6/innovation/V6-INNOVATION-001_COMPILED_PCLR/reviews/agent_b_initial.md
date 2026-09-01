# Agent B initial code review

- Reviewer: temporary Reviewer B
- Phase: stage-1 independent read-only code review
- Frozen code commit: `b707b0c4671051244cebf4f8404299fc016b281e`
- Base commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- RUN config SHA256: `73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`
- Asset manifest SHA256: `3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`
- Relation asset manifest SHA256: `0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`
- Scope: training/evaluation loop, matched online-V5 control fairness, checkpoint/output contract, 200-epoch schedule, same-checkpoint S/V/I knockouts, reproducibility, and minimal-overdesign risk.
- Constraint: I did not read any Agent A review file before writing this report. I did not edit code or run a server. I created only this review file; the parent `reviews/` directory did not exist and had to be created to place this file at the requested path.

## Files and identities read

- `AGENTS.md`
- `research/ideas/IDEA-201_compiled_pclr.md`
- `model/frameworks/v6/compiled_pclr.py`
- `model/frameworks/v6/train_compiled_pclr.py`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/EXPERIMENT.yaml`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/configs/RUN-001.yaml`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/PARAMETER_MATRIX.csv`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/result.md`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/framework_diagram.html`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/CODE_REVIEW.md`
- `experiments/v6/EXPERIMENT_QUEUE.csv`
- `experiments/v6/FRAMEWORK.yaml`
- `experiments/v6/innovation/INDEX.md`
- `tests/frameworks/v6/test_compiled_pclr.py`
- Relevant parent/control references in `model/frameworks/v4/train.py`, `model/frameworks/v4/model.py`, `model/frameworks/v4/pclr.py`, `model/frameworks/v4/evaluate_pclr_semantic_ensemble.py`, and `model/frameworks/v5/model.py`.
- Accurate diff reviewed: `52b511d77b4ad048f35b40dc3cbd9afd092167e9..b707b0c4671051244cebf4f8404299fc016b281e`.

Identity checks performed:

- `git rev-parse HEAD` returned `b707b0c4671051244cebf4f8404299fc016b281e`.
- `Get-FileHash` on `configs/RUN-001.yaml` returned `73A812268B18E9F46A2CEDF59ACDABB8EF0CDB13388EC83B5F23B73475E4239B`.
- `git diff --stat base..commit` showed the expected V6/IDEA-201/model/test additions.
- I did not rerun the provided `29 passed` suite because this phase prohibited server work and only required reading; running pytest can create local cache files. The provided test result is treated as external evidence, not re-certified by this review.

## Summary verdict

`pass`.

I found no P0 or P1 issues in the frozen code/RUN contract. The implementation is eligible for the next phase from Reviewer B's side, subject to Agent A's independent report and the required direct file-exchange response stage.

## P0

None.

## P1

None.

## P2

1. `result.md` and `CODE_REVIEW.md` still contain historical `89b2908...` independent-pass/cross-blocked text. `CODE_REVIEW.md` explicitly marks that history as superseded, but `result.md` can still be misread as current review state.
   - Closure condition: after both reviewers finish the current `b707b0c...` file-based exchange, update the experiment review/result ledger to bind the current commit, current config SHA, current review files, and final cross-response status.

2. `experiments/v6/FRAMEWORK.yaml` and `experiments/v6/innovation/INDEX.md` still describe older V6 development status and mention fixed-150/planned-pre-run wording for C-PCLR. This does not affect the RUN executable contract, which is correctly frozen in `EXPERIMENT.yaml` and `RUN-001.yaml`, but it is a documentation consistency risk.
   - Closure condition: update only the lightweight ledger after review/RUN status is known; do not change code identity for this P2 alone.

3. The formal RUN does not persist an in-progress `checkpoint_last.pth`; it writes `model_best.pth`, `evaluation_history.json`, and `metrics.json` only at successful completion. A 28,228-step run interrupted mid-flight would need to restart.
   - Closure condition: either accept full restart as the explicit Gate-B cost, or add a resume checkpoint in a future semantics-changing review. This is not a current correctness blocker because completed results remain atomically written.

4. `_finite_source_gradients()` and `_finite_control_gradients()` require finite gradients but do not require positive nonzero norms for source/control groups. This is adequate for avoiding the prior false failure on compatibility parameters and for detecting broken graph participation at the head level, but it is less strict than a full liveness proof for every source/control sub-parameter.
   - Closure condition: if future runs need stronger receipts, record per-group nonzero counts/norms while preserving the allowed `semantic_group_logits.grad is None` compatibility exception.

## Key review findings

### Training and matched-control fairness

The training source is built from the R2 source config rather than initialized from the R2 checkpoint. `load_training_source()` validates the source checkpoint file and SHA but does not load it; it builds `source = build_model(source_config, tensors, device)` and sets trainable flags for Parent, Gate, Reader, and beta. `load_parent_control()` is the only path that loads `source_checkpoint`, and it is used for read-only formal V5 parity.

The matched online-V5 control is trained in the same loop and on the same sampled batches as C-PCLR:

- Parent/Gate receive Parent CE, topology, and frozen-oracle gate loss.
- The online V5 control receives the original PCLR `relation_loss` and `beta_loss`.
- The C-PCLR head receives final 200-class CE and incident-edge direction CE.
- Parent, control, and head optimizers are stepped once per update.

The losses are summed before backward, but the gradient boundaries remain separate in the current code: PCLR relation/beta objectives do not update Parent/Gate, C-PCLR head losses do not update the shared source, and the source Parent/Gate trajectory is shared by both C-PCLR and matched online-V5. This is a fair matched-control construction for the stated contract.

### 200-epoch schedule

The config locks `nominal_epochs: 200`, `total_updates: 28228`, `batch_size: 50`, and `eval_interval_steps: 141`. The loop runs `range(1, total_updates + 1)` with independent `torch.randperm(len(train_features), generator=generator)[:50]` each step. Evaluations occur every 141 updates and at the final tail update 28,228. Teacher refresh uses the existing `teacher_refresh_updates()` convention and starts from an initial package, matching the original v4 cadence.

### Checkpoint selection and output

C-PCLR selects its own best checkpoint by maximum Full H after update 0. The matched online-V5 control separately tracks its own best Full H after update 0. The final acceptance gate requires C-PCLR Full H to exceed `max(formal V5 H, matched online-V5 best H)`.

The output writes:

- C-PCLR `best_update`, `model_state_dict`, `source_model_state_dict`, and `best_zs_observation`;
- matched online-V5 `best_update`, `best_metrics`, `best_state_dict`, and independent `best_zs_observation`;
- full `evaluation_history.json`;
- final `metrics.json` with deltas against formal V5 and matched online-V5.

The final S/V/I measurements are computed after reloading the C-PCLR best head state, so the module-off metrics use the same Full checkpoint.

### Official test and data boundary

The RUN contract correctly discloses Chen-style official-test selection:

- `test_used_for_selection: true`
- `test_used_for_hyperparameter_selection: true`
- `nested_official_test_selection: true`
- `unseen_images_used_for_gradient: false`
- `strict_blind_claim: false`

Training uses `train_features`/`train_labels` only. Evaluation uses official test seen/unseen only for metric selection/reporting. I did not find a path where official unseen images or labels enter gradient generation.

### Same-checkpoint S/V/I closure

The implementation matches the pre-registered closure semantics:

- `S-off`: `semantic_enabled=False`, role residual removed from `Q_image`.
- `V-off`: `visual_enabled=False`, Reader residual bypassed via `normalize(x)`.
- `I-off`: `interaction_enabled=False`, relation side of `Q` zeroed.

All are evaluated from the same reloaded best C-PCLR head checkpoint, with no retraining.

### Prior gradient/dropout issues

The previous source-gradient false failure is closed. `_finite_source_gradients()` now allows group-internal compatibility parameters with `grad=None`, while requiring every non-empty active group to have at least one finite actual gradient. The test `test_real_parent_receipt_allows_compatibility_parameter_without_grad()` covers the real `PaperV2ThreeModuleModel(full/off/off)` path and confirms `semantic_group_logits.grad is None` while the receipt still passes.

Prototype synchronization and head construction use temporary eval mode and restore CPU/CUDA RNG state, which closes the prior dropout/RNG contamination risk for the current uniform train/eval usage.

## Strongest counterexample

The strongest remaining counterexample is empirical, not code-structural:

If the matched online-V5 control trained for the same seed, same 28,228 updates, same Parent/Gate trajectory, and same official-test selection reaches or exceeds C-PCLR Full H, then C-PCLR has no current accuracy advantage over the fair online-relation baseline and must be dropped under the pre-registered Gate-B contract. Separately, if any same-checkpoint S/V/I-off condition reduces H by less than `1.0pp`, the deployment dependency claim fails even if Full beats the parent/control.

The code now encodes those failure conditions directly, so this counterexample is properly testable by the RUN.

## Final decision

Reviewer B stage-1 independent decision: `pass`.

P0=0, P1=0. P2 items are non-blocking and should not delay a reversible Gate-B RUN after Agent A and the required cross-response step are complete.
