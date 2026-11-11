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

## Free Hosting

The GitHub Actions workflow in `.github/workflows/deploy-pages.yml` builds the Vite site for the repository path `/Challenge-The-AI-Lead/`. Enable **Settings → Pages → Build and deployment → GitHub Actions**. The build uses `https://everyday-ai-api.onrender.com` by default; if Render assigns a different hostname, add a repository Actions variable named `EVERYDAY_AI_API_URL` with the full HTTPS origin.

For the API, create a Render Blueprint from this repository and select the `everyday-ai-api` free web service in [`render.yaml`](render.yaml). The Blueprint builds the SMS and bike artifacts locally and removes downloaded ZIP archives and private test split CSVs before runtime. CORS allows `https://jaynightmare.github.io`; for a custom Pages domain, set `EVERYDAY_AI_CORS_ORIGINS` in Render to the exact site origin.

Render Free is suitable for a workshop demo, not reliable production hosting: the service can sleep after 15 idle minutes, cold starts can take about a minute, the free web service has 512 MB RAM and an ephemeral filesystem, and free instance hours are limited. Keep organiser test labels and private-test reports out of the service. GitHub Pages is static only; all inference and CSV validation scoring are served by the separate API.

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

## Student Prediction Submissions

The fight page accepts prediction CSVs only; it never accepts joblib, pickle, PyTorch checkpoints, or executable model code. Download the task's row-ID template from the fight page, fill its `prediction` column, and upload it. The strict schema is exactly:

```csv
row_id,prediction
sms-opaque-id,ham
```

Text predictions must be `ham` or `spam`; vision predictions must be `keyboard`, `mouse`, or `headphones`; bike predictions must be finite numeric rental counts. The server requires exactly one unique prediction for every validation row ID, rejects extra/missing rows and columns, caps files at 512 KiB, scores against validation labels only, and does not save uploaded CSVs. Negative bike predictions are clamped to zero, matching evaluation. CSVs do not change the stored baseline artifacts.

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for milestones, exact data limitations, hardware notes, and the complete CLI reference. `pytest` runs the local safety and contract tests. Dataset downloads and training are not run by the API or frontend.

[`notebooks/experiment_walkthrough.ipynb`](notebooks/experiment_walkthrough.ipynb) calls the same pipeline modules. Its download and training flags default off; the private final test remains an organiser-only CLI operation.

Frontend dependency advisories were cleared by removing the unused Vitest development dependency; there are no frontend test files. The npm lockfile audit reported zero advisories after removal. Production dependencies had no advisories before the change.
