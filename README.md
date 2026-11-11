# Everyday AI: Beat the AI Lead

A local-first Tech Society showcase of three educational ML tasks. The demos are decision-support prototypes, not production services. No cloud services, paid APIs, account system, or database are used. Training can use the network only to fetch the named datasets, Python packages, and the permitted ImageNet weights; inference runs locally.

## Status

The repository begins with pipelines and a UI that report missing data or artifacts explicitly. Nothing is presented as trained until an artifact and its recorded evaluation exist. The equipment image task awaits a locally collected pilot dataset; SMS and bike data are downloaded only when their preparation commands are run. Validation metrics are the only metrics intended for the student UI. Final test evaluation is an organiser-only local command.

## Quick start

Python 3.11-3.13 and Node.js 20.19+ are required. On Debian/Ubuntu, install the matching `python3-venv` OS package first if `python -m venv` reports that `ensurepip` is unavailable. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cd frontend && npm install && cd ..
```

For notebook plots and the optional vision trainer, install `python -m pip install -e '.[dev,notebook,vision]'`; torchvision weights are fetched only when CV training is explicitly run.

Run the API and UI in separate terminals:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

Open the local URL printed by Vite. The API binds to loopback by default. Local model files are trusted organiser-generated artifacts; never place an artifact from an unknown source in `artifacts/` because scikit-learn joblib files can execute code when loaded.

## Reproduce an experiment

SMS (official UCI SMS Spam Collection, dataset ID 228; attribution and CC BY 4.0 terms are recorded in the run metadata):

```bash
python -m ml.text.cli prepare
python -m ml.text.cli train --model-id starter
python -m ml.text.cli train --model-id ai-lead
python -m ml.text.cli evaluate --model-id starter --split validation
python -m ml.text.cli evaluate --model-id ai-lead --split validation
python -m ml.text.cli evaluate --model-id starter --split test --organiser
python -m ml.text.cli summarize
python -m ml.text.cli export-student
```

Bike Sharing hourly (official UCI dataset ID 275; `cnt` is the target):

```bash
python -m ml.tabular.cli prepare
python -m ml.tabular.cli train --model-id starter
python -m ml.tabular.cli train --model-id ai-lead
python -m ml.tabular.cli evaluate --model-id starter --split validation
python -m ml.tabular.cli evaluate --model-id ai-lead --split validation
python -m ml.tabular.cli evaluate --model-id starter --split test --organiser
python -m ml.tabular.cli summarize
python -m ml.tabular.cli export-student
```

Equipment images are not downloaded or replaced with unrelated images. Collect and label them as described in [`docs/vision-data.md`](docs/vision-data.md), then run:

```bash
python -m pip install -e '.[vision]'
python -m ml.vision.cli prepare
python -m ml.vision.cli train --model-id starter
python -m ml.vision.cli train --model-id ai-lead
python -m ml.vision.cli evaluate --model-id starter --split validation
python -m ml.vision.cli evaluate --model-id ai-lead --split validation
python -m ml.vision.cli export-student
```

Student export commands copy only training and validation rows/images into ignored `data/student/`; they never copy the private final test partition. Add an organiser-owned access process before distributing those files.

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for milestones, exact data limitations, hardware notes, and the complete CLI reference. `pytest` runs the local safety and contract tests. Dataset downloads and training are not run by the API or frontend.

[`notebooks/experiment_walkthrough.ipynb`](notebooks/experiment_walkthrough.ipynb) calls the same pipeline modules. Its download and training flags default off; the private final test remains an organiser-only CLI operation.
