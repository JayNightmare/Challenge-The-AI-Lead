You are my development agent in VS Code.

PROJECT
Build “Everyday AI — Beat the AI Lead”, a local-first educational
showcase for a university Tech Society.

The application demonstrates three practical machine-learning tasks:

1. Computer vision: equipment-category recognition.
2. NLP: SMS spam classification.
3. Tabular regression: bike-rental demand estimation.

- These are separate educational prototypes presented through one interface.
- They are not production services.
- Store each dataset, trained model, and run metadata locally in the designated directories.
- Pool all local resources to avoid redundant downloads and ensure reproducibility.
- Access to the internet should be limited to downloading datasets and dependencies only.
- Access all three models to operate locally without relying on external services.
- Ensure all models can be accessed through a web interface for students to interact with and learn from.
- Include a benchmark comparison for each model as a part of the evaluation. Make it a "YOU vs AI Lead" comparison drag and drop the model into the "fight" interface.

PRIMARY OBJECTIVE
Deliver three reproducible model pipelines and a usable demonstration UI.
Each challenge must compare:

- A simple starter baseline.
- A stronger “AI Lead” baseline.

Do not invent performance results or assume the stronger model wins.
Measure results and report them honestly.

STACK

- Frontend: React, TypeScript, Vite, Tailwind.
- Backend: Python, FastAPI, Pydantic.
- ML: PyTorch/torchvision and scikit-learn.
- Tests: pytest and frontend tests where useful.
- Local files for datasets, artifacts, and run metadata.
- No database, authentication, cloud deployment, paid APIs, or LLM API.
- CPU inference; optional CUDA acceleration for CV training.
- Verify current dependency compatibility before choosing versions.

FIRST ACTIONS

1. Inspect the existing workspace. Do not overwrite existing work.
2. Create a concise implementation plan and repository structure.
3. Document unresolved dataset and hardware requirements.
4. Begin implementation in small, testable milestones.
5. Clearly report anything blocked by missing data or unavailable downloads.

SUGGESTED STRUCTURE
frontend/
backend/
ml/
  common/
  vision/
  text/
  tabular/
configs/
notebooks/
data/
artifacts/
tests/
docs/

Keep reusable ML code in Python modules.
Notebooks should call those modules rather than duplicate the pipelines.
Git-ignore raw data, trained weights, and private evaluation files.

DEMO 1: EQUIPMENT CHECK-IN ASSISTANT

Use case:
Suggest a category for a photographed equipment item so a human can
confirm it during check-in.

Initial classes:

- keyboard
- mouse
- headphones

Scope:

- One main object per image.
- Image classification, not object detection.
- Do not claim to identify individual assets, damage, or ownership.
- Do not automatically update any inventory.

Models:

- Starter: frozen ImageNet-pretrained ResNet-18 with a trained classifier.
- AI Lead: the same permitted pretrained weights, with controlled
  fine-tuning and training-only augmentation.

Data:

- Implement an image-folder/manifest ingestion pipeline.
- Provide instructions for collecting a small pilot dataset.
- Record class, physical object ID, and collection session.
- Split by physical object where possible, not randomly by adjacent frames.
- Check for duplicate files and overlapping object IDs between splits.
- Warn when there are too few distinct objects to assess generalisation.
- Fit nothing using validation or test data.
- Never silently substitute an unrelated dataset.
- If images are unavailable, implement the complete pipeline but mark the
  task “awaiting dataset”, not “trained”.

UI:

- Upload an image or capture a single webcam frame.
- Show suggested category, class scores, model version, and inference time.
- Explain that scores do not guarantee correctness.
- Warn that unfamiliar objects can still receive a known-class prediction.
- Low-score warning thresholds must be configurable and selected on
  validation data; do not call them reliable unknown-object detection.
- Do not save uploaded images by default.

Metrics:

- Macro-F1, accuracy, per-class precision/recall, confusion matrix.
- Show representative mistakes using approved dataset examples.

DEMO 2: SMS SPAM SCREENING ASSISTANT

Use case:
Suggest whether an SMS resembles spam for human review.

Dataset:
UCI SMS Spam Collection.
Use the official repository as the source and record attribution/license.
Verify the download and parser against the actual source format.

Models:

- Starter: TF-IDF word features plus Multinomial Naive Bayes.
- AI Lead: TF-IDF word/character features plus logistic regression.
- Select configurations using training/validation data only.
- Do not introduce a large transformer in the first implementation.

Data:

- Detect duplicates and prevent duplicate text crossing dataset splits.
- Use stratification where compatible with grouping.
- Fit vectorisers only on training data using sklearn Pipelines.
- Keep a fixed split manifest and seed.
- Do not expose raw dataset messages through the demo UI.
- Provide clearly fictional example messages for demonstrations.

UI:

- Text entry and fictional example buttons.
- Spam/not-spam prediction with the appropriate score label.
- Clearly distinguish a model score from a calibrated probability.
- No automatic deletion, blocking, or security assurances.
- Explain that this is not a phishing detector and may not generalise
  to modern messages.
- Do not store entered messages or log request bodies.

Metrics:

- Spam precision, spam recall, spam F1, macro-F1, confusion matrix.
- Explain false positives and false negatives.
- Evaluate majority-class performance as a reference.

DEMO 3: BIKE-RENTAL DEMAND ESTIMATOR

Use case:
Estimate hourly total rental demand from calendar and weather inputs
to illustrate resource-planning support.

Dataset:
UCI Bike Sharing, hourly data.
Verify the official source, license, attribution, units, and field meanings.

Target:
cnt — total rentals.

Models:

- Starter: training-only mean demand grouped by hour and working-day flag,
  with a training-only global-mean fallback.
- AI Lead: HistGradientBoostingRegressor with appropriate preprocessing.
- Explain that a tabular model is not necessarily a neural network.

Leakage prevention:

- Never use casual or registered as input features: they sum to cnt.
- Exclude row identifiers.
- Use chronological train/validation/test partitions.
- Use time-respecting validation for tuning.
- No random train/test split.
- Use only fields available under the explicitly documented prediction
  scenario.

UI:

- Controls for hour, calendar variables, and weather inputs.
- Display units and convert physical inputs to dataset encodings correctly.
- Show predicted rental count and baseline comparison.
- Label it as a historical scenario estimate.
- State that observed weather is not the same as a weather forecast.
- Warn that arbitrary scenarios can be outside the training distribution.
- Do not claim London-specific validity or causal effects.
- Do not fabricate prediction intervals.
- Keep any non-negative output handling explicit and consistent in evaluation.

Metrics:

- MAE and RMSE.
- Predicted-versus-actual chart and errors by hour.
- Plain-language explanation of errors in rental-count units.

SHARED TRAINING AND EVALUATION

Provide CLI commands for:

- Data preparation and validation.
- Starter model training.
- AI Lead model training.
- Validation evaluation.
- Separate final-test evaluation.
- Generating the frontend-readable metrics summary.

Every trained artifact must record:

- Task and model ID.
- Dataset source/version or checksum.
- Split manifest.
- Configuration and random seeds.
- Dependency versions.
- Training timestamp and available hardware.
- Metrics with explicit split names.

Save preprocessing with each model.
Check that API inference uses the same preprocessing as evaluation.

Separate organiser and student workflows:

- Student materials include training and validation data only.
- Private test labels stay outside the frontend and public repository.
- Final evaluation is a local organiser CLI, not a public upload endpoint.
- The UI must not repeatedly reveal private-test feedback.
- Do not load untrusted pickle/joblib/model submissions.

Record a fixed baseline artifact for each challenge.
Do not secretly replace it as members improve.

FRONTEND

Build:

1. Overview page introducing the three use cases.
2. A page for each live demo.
3. A baseline-comparison/results page.
4. A “How to beat this model” section with experimental ideas.
5. A limitations and data-sources section.

Use a clean navy-and-off-white style with accessible contrast.
Make loading, validation, missing-data, and missing-model states explicit.
No fake leaderboard, fabricated accuracy, or simulated successful training.

When an artifact is missing:

- Show “Model not trained”.
- Explain the next command needed.
- Do not return mock predictions as real inference.

BACKEND

Provide typed endpoints for:

- Health/model readiness.
- Model metadata and approved metrics.
- Image classification.
- Text classification.
- Demand estimation.

Validate file types, image dimensions, payload sizes, input ranges,
and empty text. Configure local-only CORS.

Load artifacts once at startup rather than per request.
Do not execute training inside inference requests.
Do not accept arbitrary artifact paths from clients.

TESTS

Cover:

- Schemas and invalid inputs.
- Missing/corrupt model handling.
- Split overlap and duplicate checks.
- Exclusion of target-leaking bike features.
- Chronological split ordering.
- Train/inference preprocessing consistency.
- Artifact save/load round trips.
- Prediction endpoint contracts.
- No user-input persistence by default.

Synthetic fixtures may be used in tests only.
Never present synthetic test results as real dataset performance.

DELIVERY MILESTONES

1. Scaffold the repo, APIs, frontend shell, and readiness states.
2. Implement and genuinely train/evaluate the SMS prototype.
3. Implement and genuinely train/evaluate the bike-demand prototype.
4. Implement CV ingestion, collection guidance, training, and inference.
   Train it only when suitable images are available.
5. Connect all available artifacts to the UI.
6. Add notebooks, experiment templates, tests, and documentation.

After each milestone:

- Run relevant tests.
- Report what works, what was actually executed, and what remains blocked.
- Include exact setup/run commands.
- Never describe an unexecuted training run as completed.

ACCEPTANCE CRITERIA

- One interface supports three genuine ML use cases.
- Each available model has reproducible training and measured evaluation.
- Starter and AI Lead results are compared honestly.
- The CV data dependency is explicit.
- Inference works locally without paid APIs once dependencies and artifacts
  are available.
- A beginner can follow the README to reproduce one complete experiment.
