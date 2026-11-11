# Implementation Plan and Status

## Structure

```text
backend/app/       FastAPI contracts, readiness, local inference
ml/common/         split, artifact metadata, and metric utilities
ml/text/           UCI SMS preparation, baselines, evaluation
ml/tabular/        UCI hourly bike preparation, baselines, evaluation
ml/vision/         equipment manifest, object split, ResNet pipelines
frontend/          React, TypeScript, Vite, Tailwind local UI
configs/           fixed seeds and task configuration
data/raw/          ignored source downloads / organiser images
data/processed/    ignored local split files
artifacts/         ignored trusted local model files and metadata
reports/           ignored local evaluations
tests/             safety and contract tests
notebooks/         thin experiment entry points calling ml modules
```

## Milestones

1. Scaffold typed local API, frontend readiness states, and repository guidance.
2. Implement and measure SMS starter and AI Lead models on fixed, duplicate-grouped splits.
3. Implement and measure bike baselines on chronological splits with target-leakage exclusions.
4. Implement equipment image ingestion, object-grouped splits, ResNet training, evaluation, and inference; train only with suitable collected images.
5. Connect only locally available artifacts and validation summaries to the UI; retain test evaluation as organiser-only.
6. Add tests, experiment notebooks, collection guidance, and reproducibility documentation.

The initial workspace contained only `Mission.md`. No data, artifacts, tests, or dependency lockfiles were present. Implementation status is therefore based on code and test execution, not assumed training results.

## Delivery Status

- Milestone 1 complete: local FastAPI readiness API, React/Vite/Tailwind shell, explicit missing-data/model states, and repository guidance.
- Milestones 2-3 complete: UCI SMS and bike data were downloaded and parsed; starter and AI Lead models were trained, validation-evaluated, and run once through the organiser-only test CLI. Student exports contain training and validation rows only.
- Milestone 4 pipeline complete, training blocked: image manifest ingestion, duplicate checks, object-disjoint splitting, ResNet training/evaluation, upload inference, and collection instructions are implemented. No pilot images were present, so no CV model or metrics were produced.
- Milestone 5 complete for trained text/bike artifacts: API startup loading, local inference, validation summary, benchmark UI, and strict CSV-only validation submissions are connected. Vision remains awaiting its dataset.
- Milestone 6 complete for tests, docs, and notebook walkthrough. The notebook's safe default path was executed; its download/training flags remain off.

Measured validation results are recorded in ignored local reports and the frontend summary. On this run SMS macro-F1 was 0.858 (starter) and 0.950 (AI Lead); bike MAE was 125.9 and 45.3 rentals/hour respectively. Results are specific to these fixed splits and do not imply future performance.

## Unresolved requirements and blockers

- **Equipment images:** no pilot images exist yet. Collect diverse photos of multiple physical keyboards, mice, and headphones across sessions, backgrounds, distances, and lighting. Keep object IDs in the manifest; fewer than three distinct objects per class prevents a meaningful object-disjoint train/validation/test split. The CV challenge remains awaiting dataset until this is satisfied.
- **SMS and bike downloads:** source archives are fetched on explicit preparation commands. A successful download and source-format validation must precede training. The UI never exposes raw SMS examples.
- **CV training hardware:** CPU is supported but may be slow. CUDA is optional; record the actual device in run metadata. Inference is CPU-compatible.
- **Pretrained weights:** torchvision's approved ImageNet ResNet-18 weights are fetched only when training is requested and are cached by torchvision. No weights are silently substituted.
- **Final test privacy:** local test split files and labels are ignored by Git. Never publish them or route test metrics to the student UI. Student-distributed data must contain training and validation rows only.
- **Free hosting:** GitHub Pages serves only the Vite static bundle. The Render Blueprint trains text/bike artifacts during build, removes test CSVs and dataset ZIPs before runtime, and keeps validation-only scoring in the API. Render Free sleeps when idle and uses ephemeral storage; it is for a workshop demo, not a production availability promise.

## Data sources and task limits

- SMS Spam Collection: UCI Machine Learning Repository, dataset 228, `https://archive.ics.uci.edu/dataset/228/sms+spam+collection` (CC BY 4.0; DOI `10.24432/C5CC84`). The downloader checks the archive and parser expects the source's tab-separated label/message rows. Attribution and license metadata are written beside prepared data. Predictions are not phishing detection or a security guarantee.
- Bike Sharing: UCI Machine Learning Repository, dataset 275, `https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset` (CC BY 4.0; DOI `10.24432/C5W894`). The downloaded archive's `Readme.txt` defines `temp` as Celsius divided by 41, `atemp` as Celsius divided by 50, humidity as percent divided by 100, and wind speed as its dataset value divided by 67. The current UCI web variable table instead describes temperature as min-max scaled and lists different season names than the bundled archive readme; the app uses the downloaded archive's actual normalization and shows numeric season codes to avoid silently choosing a conflicting season label. The hourly `cnt` target is total rentals; `casual`, `registered`, and `instant` are excluded. Inputs represent an explicitly supplied historical scenario, not a forecast or causal estimate.
- Equipment: locally collected and consented Tech Society pilot images only. One primary object per image; no identification, damage assessment, ownership claim, or automatic inventory updates.

All three tasks are educational prototypes and may fail on distribution shifts. Scores are not guarantees. Low-score thresholds, if enabled, are validation-selected warnings only and do not provide unknown-object detection.

Student submissions use an exact `row_id,prediction` CSV schema. IDs are opaque and generated separately from model features. The API checks exact validation row coverage, class labels or finite numeric values, and file size; it does not persist the CSV or load arbitrary artifacts. The final test partition is not available through these endpoints.
