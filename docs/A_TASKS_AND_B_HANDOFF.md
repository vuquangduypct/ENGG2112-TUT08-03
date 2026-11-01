# Role A tasks and handoff to Role B

Updated 1 November 2026. Original private dataset version: `aptos5-provisional-v1`.

**Main interactive deliverable:** [ENGG2112_Data_Engineering.ipynb](../jupyter%20notebooks/ENGG2112_Data_Engineering.ipynb) combines the actual code, recorded findings and handoff instructions. Use inspect mode with the existing private handoff, or build mode with authorised originals and a new run name. Existing scripts and report exports remain available.

**Current status:** the data interface is ready for integration and initial
training/validation experiments. Duplicate review is incomplete, so this is not
a frozen dataset for final evaluation. Model notebooks now contain saved validation results; this does not establish final test performance. See [current status](PROJECT_STATUS.md).

## Role A responsibilities

Role A prepares the data, explains the project background and data decisions,
and supplies reproducible inputs to Role B. Role B develops and trains the models.
Role C leads evaluation discussion and final report assembly, working with A and B.

| Deliverable | Current status | Evidence or location |
|---|---|---|
| Original five-grade task | Implemented; team agreement still required | Grades 0 to 4 in `src/common.py` |
| Source and local file checks | Completed for the recorded dataset | `src/audit_data.py`; English data report |
| Class distribution | Completed | `docs/figures/class_distribution.png` |
| Exact duplicate and conflict detection | Completed | Audit script; 62 records provisionally excluded |
| Review of similar images | Outstanding | Local `artifacts/audit/near_duplicate_candidates.csv` |
| Provisional grouped split | Generated | 2,521 training, 538 validation, 541 test records |
| Deterministic image preparation | Completed baseline | 224x224 RGB, resize and black padding |
| Reusable loading interface | Completed | `src/handoff_data.py`; `examples/load_data_for_b.py` |
| English report and instructions | Prepared | `reports/`, `README.md`, `B_START_HERE.txt` |
| Background and primary literature | Outstanding | `docs/data_engineering_notes.md` |
| Full AI-use appendix | Outstanding | `docs/AI_ASSISTANCE.md` describes what to retain |
| B-side execution | Saved model runs demonstrate data loading and training; formal checksum acceptance still needs recording | Model Notebook outputs; acceptance checklist below |

## What is in GitHub and what is shared separately

The public contribution includes English code, tests, a Notebook with empty saved
outputs, dependency files, aggregate figures, the data report, and these handoff
instructions. The repository contained three CSV files before this contribution;
this change neither adds nor alters them.

Original and prepared retinal images, per-image audit records, split manifests and
the private archive are not added by this contribution. Each teammate must obtain
authorised access through the [APTOS competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
Use an authorised team channel for the private handoff. A GitHub clone alone is
not a copy of the training data.

The private handoff `ENGG2112_B_private_data_v1.zip` contains 3,600 prepared images,
the original label CSV, split manifests, excluded records, preprocessing settings,
checksums, code and reports. It excludes the full-resolution originals. The report
describes the 7 October analysis; the package was prepared on 13 October.

## Instructions for B

1. Read `B_START_HERE.txt`, then obtain the private archive from A after confirming
   authorised dataset access. Extract into a new folder. The prepared-image loading
   example works with that archive; the full audit Notebook also needs originals.
2. Create a Python 3.12 environment. See README for installation. A's exact dependency
   versions are recorded in `requirements-lock.txt`; the training framework is separate.
3. Run `python -m src.handoff_data --verify` from the extracted project root. It should
   validate 3,600 prepared images with split counts 2,521 / 538 / 541.
4. Run `python -m examples.load_data_for_b`. It should load an RGB array with shape
   `(224, 224, 3)`, dtype `uint8`, and an integer grade in 0 to 4.
5. Connect the shared manifest to the model's data loader by `id_code`. Do not derive
   labels from directory order or make another random split.
6. Confirm the model/weights, framework, image resolution, channel order and
   normalization. The delivered pixels remain in 0..255; apply the model's required
   transform exactly once. Random augmentation or balancing belongs only in training.
7. Reserve the test set. The helper blocks test access by default. An explicit override
   does not mean duplicate review is complete or final evaluation is authorised by the team.

If B needs higher-resolution inputs, regenerate them from the authorised original
images. Upscaling 224-pixel images cannot restore detail discarded by resizing.

## Acceptance record to complete with B

Copy this checklist into the team's experiment record and fill it after B runs the code.
Do not tick items based solely on the tests performed on A's computer.

- [ ] B confirms the original five-grade prediction target.
- [ ] B has authorised dataset access and has obtained the matching private version.
- [ ] Verification passes on B's computer with 3,600 images and the expected counts.
- [ ] The reading example returns an image and the correct associated label.
- [ ] Model and pretrained weights are recorded.
- [ ] Input resolution, channel order, normalization and augmentation are recorded.
- [ ] A and B agree on duplicate/conflict handling and the reviewed split version.
- [ ] The reviewed split is frozen before final model evaluation.
- [ ] The team agrees on per-grade and overall metrics and discussion of ordinal errors.

Record the date, handoff version, split-manifest hash, Python/library versions,
model settings and any verification errors with this acceptance record.

## Work A still needs to complete

### Review duplicates and document the decision

The audit found 123 exact pixel groups containing 251 records. Thirty groups have
conflicting grades and account for 62 records. Those records remain provisionally
excluded; no diagnosis was guessed or corrected. Document whether exclusion is
retained and why. Consider whether same-grade duplicates should remain repeated
records within a set.

Review the 4,096 near-duplicate candidate pairs as candidates, not proven duplicates.
Prioritise candidate pairs that currently cross split boundaries and record confirmed
related-image groups. Visual similarity alone does not establish patient identity.
Use `data/reviewed_groups.csv` and the README command to generate a reviewed split.
Generate a new version and matching prepared-image manifests if assignments change.

### Write the background with evidence

Read primary sources on diabetic retinopathy screening, fundus imaging and relevant
image-based classification methods. Explain the intended application, why these data
fit the problem, and why the revised task retains five grades. Record each source
against the specific claim it supports. Do not assert elderly-subgroup performance,
clinical benefit or population prevalence without appropriate evidence.

### Complete the final report contribution

Combine the background and citations with the data source, class distribution,
integrity checks, duplicate decisions, preprocessing rationale and actual final split.
Revise the existing report when review or preprocessing choices change. Label the
current split as provisional until that work is complete. Coordinate the final section
length with the person assembling the group report.

### Preserve the AI-use record and explain the work

Retain relevant original prompts and outputs, student modifications, verification
results and reasoning for the course appendix. The short repository disclosure does
not replace that record. Be able to explain the pipeline and its assumptions in your
own words.

## Verification already performed

The complete Notebook has 28 cells, including 16 code cells. Its JSON structure
was validated, and all code cells executed in Python 3.12 against the real
3,600-image handoff. A separate synthetic-data run checked a fresh build,
conflicting-label exclusion, group separation, standalone B-side inspection,
reserved-test access and refusal to overwrite an existing run. Published code
cells have no saved outputs. These checks do not train or evaluate a model.

The project has 12 passing automated tests covering invalid labels, duplicate grouping,
reproducible splits, image transforms, ID-based joins, mismatched labels, unsafe paths,
missing images and reserved test access. All eight analysis Notebook code cells were
executed successfully against A's local data. All 3,600 prepared images were decoded
and checked against their handoff hashes. The private archive was also extracted into
a different folder and its verifier and reading example succeeded.

These checks establish software and package consistency. They do not validate medical
labels, establish patient independence or substitute for running the handoff on B's
computer.
