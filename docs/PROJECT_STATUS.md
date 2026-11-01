# Project status and evaluation evidence

Updated 1 November 2026.

## Completed and recorded

- The data workflow audits files and labels, detects exact duplicates and label conflicts, creates grouped provisional splits, and prepares 224x224 RGB images by aspect-preserving resizing and black padding.
- The original recorded audit excluded 62 conflicting-label records, leaving 3,600 images (2,521 train, 538 validation, 541 reserved test).
- All three model notebooks contain saved validation outputs. ResNet-18 is fine-tuned end to end; Random Forest and CNN-BiLSTM use frozen ImageNet ResNet-18 features. Model-side normalization uses ImageNet mean/std.
- The model comparison checks the split-manifest fingerprint and validation sample counts. It explicitly distinguishes a single ResNet-18 seed from three seeds for the alternative methods. This is a pipeline comparison, not a controlled architecture-only ablation.
- Saved outputs demonstrate that B loaded data and ran training. They do not independently establish that every private handoff checksum was verified.

## Recorded validation results

| Pipeline | Runs | Accuracy | Macro-F1 | Balanced accuracy | QWK |
|---|---:|---:|---:|---:|---:|
| Fine-tuned ResNet-18 | 1 | 0.8216 | 0.6607 | 0.6486 | 0.8754 |
| Random Forest on frozen CNN features | 3, mean | 0.7683 | 0.4373 | 0.4441 | 0.7965 |
| CNN-BiLSTM on frozen CNN features | 3, mean | 0.7627 | 0.6064 | 0.6547 | 0.8187 |

These figures come from the saved Notebook outputs, not a rerun of the maintenance changes. The comparison Notebook includes the alternative models' standard deviations. Its per-grade table uses the predetermined primary seed, not the three-seed average. Keep those summaries distinct in the report.

ResNet-18's recorded grade-4 recall is 15/40 (37.5%), so overall accuracy does not establish strong performance for every grade. No final test performance or clinical time saving is established by these results.

## Data versions and reproducibility

`docs/handoff_contract.json` describes the original private `aptos5-provisional-v1` package. Preserve that historical contract: changing its hashes without rebuilding and verifying the package would invalidate its purpose.

The saved model comparison records a separate generated handoff under `artifacts/notebook_runs/notebook_run_001` with split SHA-256 `81283b51d8274a7d4ee5cd873cfc6479fd9efd178a75bacd45e297dd41d57a54`. This differs from the original package's recorded split hash. Equal record counts do not establish identical manifests. Use the contract and manifests belonging to the actual experiment; do not label these as the same version without comparing the records.

Model notebooks now resolve paths from the repository root. Set `DATA_ROOT` explicitly when using a different private handoff. RF and CNN-BiLSTM require `src/model_support.py` from the same checkout. Their cache checks include actual train/validation image hashes, configuration, transforms, encoder/library versions, IDs, labels, shape and finite features. Old caches will be rebuilt; test images are not opened for cache validation.

The public repository does not include the private images or generated model result JSON/checkpoints. To rerun the comparison, run the three model notebooks first with the authorised data and matching split. Retained outputs are historical evidence, not proof that a fresh clone can run without data. Record the environment and export the new aggregate results after rerunning.

## Outstanding evaluation work

1. **A and B:** review flagged similar-image pairs, prioritising pairs that cross split boundaries; record confirmed related groups and uncertain cases. Similarity is not proof of patient identity.
2. **A and B:** document treatment of repeated images and conflicting labels, then freeze a versioned split. If assignments change, regenerate matching prepared manifests and repeat training/validation on that version.
3. **B and C:** finalise model selection using validation only, then perform the reserved-test evaluation with the selected configuration after review and freeze. Do not enable the test switch merely to produce a score.
4. **Team:** report the actual results, per-grade errors, unresolved patient-independence limitations and the distinction between validation and test results. Reconcile the final report with the actual experiment version.

The test switches remain `False`. No manual review, patient-level independence, final test run, or clinical deployment benefit is claimed by the maintenance fixes. Final report completion must be checked in the team's shared document; the historical data reports are not a substitute for that document.
