# ENGG2112: five-grade retinal image classification

Local data-engineering contribution for the APTOS 2019 project. The task uses the original five grades: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR. This revises the binary target in the submitted proposal; confirm the shared target and evaluation with the team.

## Current status

Three model notebooks contain saved validation runs; final test evaluation is still locked. See [project status](docs/PROJECT_STATUS.md) for evidence, limitations, data-version distinctions and the remaining evaluation steps. Historical model outputs are retained and explicitly labelled; maintenance fixes do not imply retraining.

## Start here

Open [ENGG2112_Data_Engineering.ipynb](jupyter%20notebooks/ENGG2112_Data_Engineering.ipynb) for the complete English data-engineering contribution in one Jupyter Notebook. It includes the actual Python implementations, audit and class analysis, grouped splitting, image preprocessing, verified loading, recorded findings, and A-to-B instructions. It does not depend on importing `src`.

Notebooks locate the project root from the repository or a prepared handoff, even when started in `jupyter notebooks/`. Set an absolute `PROJECT_ROOT` if you keep the Notebook elsewhere; set `DATA_ROOT` in model notebooks to your chosen prepared handoff.

Select the Python environment with the project dependencies, edit `PROJECT_ROOT` if needed, and run cells from top to bottom. Use `MODE = 'inspect'` with the existing private handoff. For a fresh reproduction from authorised originals, use `MODE = 'build'` and a new `RUN_NAME`; outputs go into a separate `artifacts/notebook_runs/<RUN_NAME>/` handoff. The Notebook refuses an existing build directory. See its first section for the required files. A GitHub clone alone cannot run the data cells without local data.

The [model-training notebook](jupyter%20notebooks/ENGG2112_Model_Training.ipynb) is a separate PyTorch and scikit-learn ResNet-18 baseline. It consumes the prepared handoff and existing split, selects on validation macro-F1, and keeps test evaluation locked until the split is reviewed and frozen. It cannot produce results from a GitHub clone without the authorized image data and prepared manifests. PyTorch and TorchVision are separate from the data-engineering requirements; follow the official PyTorch install selector for the chosen machine and kernel.

The [Random Forest notebook](jupyter%20notebooks/ENGG2112_Random_Forest.ipynb) and [CNN-BiLSTM notebook](jupyter%20notebooks/ENGG2112_CNN_BiLSTM.ipynb) train those methods separately using the same fixed split and cached pretrained image features. Both use `src/model_support.py` to reject caches with changed images, preprocessing, transforms, labels, IDs, or encoder versions; old caches are regenerated using train/validation images only. The [model comparison notebook](jupyter%20notebooks/ENGG2112_Model_Comparison.ipynb) compares their saved validation results with the fine-tuned ResNet-18 baseline, verifies that all runs used the same split-manifest hash, and does not train models itself. Test evaluation remains locked pending review and split freeze.

The original `.py` modules remain available for command-line use and training integration. PDF/DOCX reports are supplementary exports. Clear saved outputs before sharing a rerun Notebook.

Read [A tasks and B handoff](docs/A_TASKS_AND_B_HANDOFF.md) for completed work, remaining responsibilities and acceptance checks. The [English data report](reports/ENGG2112_Data_Analysis_Report.pdf) describes the measured provisional results.

The pipeline reads `data/train.csv` and `data/train_images/`; a fresh clone does not contain the required images or derived outputs. The root CSV files are separate project files, not substitutes for the authorised APTOS inputs. See `B_PRESENTATION_GUIDE.txt` for the handoff demonstration and file locations.

Create an environment on your computer. On Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records the exact environment used for the recorded split. For that pinned environment, install with `python -m pip install -r requirements-lock.txt`. The broader `requirements.txt` supports other environments, but package changes may change results; check the shared manifest fingerprint before comparing experiments. Dependencies here cover data work, not the teammate's CNN training framework.

## Local input data

Download data from the official [APTOS competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) after accepting its terms. Retain these files locally:

```text
data/
  train.csv                 # id_code and original diagnosis 0..4
  train_images/             # one PNG per id_code
  train_images.zip          # optional original archive
```

Do not upload the images, labels, derived images, per-image manifests, or Notebook outputs containing source images to a public repository. Each teammate should obtain authorised access directly. `.gitignore` excludes local data, artifacts and image/CSV files. See the official [competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules).

Before sharing a rerun Notebook, clear its saved outputs locally:

```powershell
.\.venv\Scripts\python.exe -m src.clear_notebook_outputs "jupyter notebooks/ENGG2112_Data_Engineering.ipynb"
```

The repository contains code and aggregate reports. Obtain dataset files separately through authorised access; do not commit private handoff archives.

## Reproduce the data work

Run from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.audit_data
.\.venv\Scripts\python.exe -m src.split_data
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

1. **Audit** validates IDs/grades, PNG integrity/full decoding, image dimensions and exact file/pixel hashes. It writes inventories, class counts, quality indicators and local figures to `artifacts/audit/`. A 64-bit dHash flags potential similar pairs for review; it is not a clinical or patient-identity test.
2. **Split** quarantines unreadable/missing records and exact pixel duplicates with inconsistent labels. Other exact pixel duplicates stay together. It searches reproducible group splits for approximately 70/15/15 proportions and similar five-grade distributions. Actual counts are reported. These splits are **provisional** until duplicate/related-image review is complete.
3. **Preprocess** writes a separate 224x224 RGB PNG baseline to `data/processed_224_v1/`. It preserves aspect ratio and pads with black. No augmentation or model-specific normalization is baked in. Automatic black-border cropping is OFF by default. Re-running into an occupied output directory is refused; choose a new `--output` folder.

Optional reviewed groups, stored locally as `data/reviewed_groups.csv`, have `id_code,reviewed_group` columns. Every image in a confirmed related set has the same reviewed_group. Exact duplicates are unioned with these groups. Then run:

```powershell
.\.venv\Scripts\python.exe -m src.split_data --reviewed-groups data/reviewed_groups.csv --output artifacts/splits_reviewed
```

For a different image size or optional black-border experiment, use a new output folder:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess --size 512 --crop-black --output data/processed_512_crop_trial
```

Compare deterministic preprocessing choices on training/validation data. Do not choose them based on final test performance. Learned normalization statistics must be fit on training data only; prescribed pretrained-model normalization belongs with the model code. Augmentation and balancing are training-only decisions for the team.

## Handoff to the model teammate

The packaged provisional handoff starts at `B_START_HERE.txt`. Run
`python -m src.handoff_data --verify` after extracting the private data pack, then
`python -m examples.load_data_for_b`. Verification checks the frozen handoff files;
it does not complete manual duplicate review. The code-only pack needs data separately.

- Read the shared `split_manifest.csv` once (currently provisional); use `id_code`, `diagnosis`, `split`, `group_id`. Do not make an independent random split.
- The processed manifest records image filenames; `preprocessing_config.json` records transformations. PNG values remain integers 0..255. Convert/normalize according to the selected model exactly once.
- The target is a five-class integer label. A binary 0/1 classifier does not implement the revised target.
- Model notebooks record the implemented methods and input settings: 224x224 RGB with ImageNet normalization. Saved validation results exist for fine-tuned ResNet-18, Random Forest on frozen CNN features, and CNN-BiLSTM. Final test evaluation remains pending review and split freeze.
- Keep test data out of tuning. Absence of exact duplicate overlap does not establish patient-level independence.

## Report and human review

Read `docs/data_engineering_notes.md` and `docs/AI_ASSISTANCE.md`. Use the locally measured audit numbers; do not copy proposed preprocessing or hypothetical outcomes as completed results. Research the disease-specific motivation and clinical literature separately. These files are project documentation, not a completed assessment submission.
