import { useEffect, useRef, useState } from "react";
import {
	ArrowDownRight,
	ArrowRight,
	ArrowUpRight,
	Camera,
	Check,
	ChevronDown,
	CircleAlert,
	Clock3,
	Cpu,
	Database,
	Download,
	FileImage,
	FlaskConical,
	Gauge,
	Info,
	LoaderCircle,
	LockKeyhole,
	Play,
	ShieldAlert,
	Sparkles,
	Upload,
	X,
} from "lucide-react";

type TaskId = "vision" | "text" | "tabular";
type PageId = "home" | TaskId | "results" | "sources";
type ModelId = "starter" | "ai-lead";
type FighterId = ModelId | "student";

type TaskReadiness = {
	status: string;
	available_models: string[];
	next_command: string;
};
type Health = { status: string; tasks: Record<TaskId, TaskReadiness> };
type TaskMetadata = {
	available_models: {
		model_id: string;
		metadata: Record<string, unknown>;
	}[];
	validation_metrics: Record<
		string,
		{
			model_id: string;
			metrics: Record<string, unknown>;
			split: string;
			representative_mistakes?: {
				row_index: number;
				expected: string;
				predicted: string;
			}[];
		}
	>;
};
type Metadata = { tasks: Record<TaskId, TaskMetadata> };
type Prediction = Record<string, unknown>;

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(
	/\/$/,
	"",
);
const apiUrl = (path: string) => `${API_BASE_URL}${path}`;

const tasks: {
	id: TaskId;
	number: string;
	name: string;
	eyebrow: string;
	summary: string;
	metric: string;
	symbol: string;
}[] = [
	{
		id: "vision",
		number: "01",
		name: "Equipment check-in",
		eyebrow: "COMPUTER VISION",
		summary: "Suggest a category for a single photographed item.",
		metric: "Macro-F1 · accuracy",
		symbol: "CV",
	},
	{
		id: "text",
		number: "02",
		name: "SMS screening",
		eyebrow: "NATURAL LANGUAGE",
		summary: "Flag messages that resemble spam for human review.",
		metric: "Spam precision · recall",
		symbol: "NLP",
	},
	{
		id: "tabular",
		number: "03",
		name: "Bike demand",
		eyebrow: "TABULAR REGRESSION",
		summary: "Estimate hourly rentals from calendar and weather inputs.",
		metric: "MAE · RMSE in rentals",
		symbol: "TAB",
	},
];

const fictionalMessages = [
	{
		label: "Expected message",
		text: "Hi, the study group starts at 4 in the library. See you there!",
	},
	{
		label: "Fictional promo",
		text: "Fictional offer: claim your sample prize by replying DEMO to 00000.",
	},
];

const emptyMetadata = (): Metadata => ({
	tasks: {
		vision: { available_models: [], validation_metrics: {} },
		text: { available_models: [], validation_metrics: {} },
		tabular: { available_models: [], validation_metrics: {} },
	},
});

function useLocalState() {
	const [page, setPage] = useState<PageId>(
		() => (location.hash.replace("#", "") as PageId) || "home",
	);
	const [health, setHealth] = useState<Health | null>(null);
	const [metadata, setMetadata] = useState<Metadata>(emptyMetadata);
	const [apiError, setApiError] = useState("");

	useEffect(() => {
		const onHashChange = () =>
			setPage(
				(location.hash.replace("#", "") as PageId) ||
					"home",
			);
		window.addEventListener("hashchange", onHashChange);
		return () =>
			window.removeEventListener("hashchange", onHashChange);
	}, []);

	useEffect(() => {
		let active = true;
		const refresh = async () => {
			try {
				const [healthResponse, metadataResponse] =
					await Promise.all([
						fetch(apiUrl("/api/health")),
						fetch(apiUrl("/api/metadata")),
					]);
				if (!healthResponse.ok || !metadataResponse.ok)
					throw new Error(
						"The local API did not respond.",
					);
				const [healthValue, metadataValue] =
					await Promise.all([
						healthResponse.json(),
						metadataResponse.json(),
					]);
				if (active) {
					setHealth(healthValue);
					setMetadata(metadataValue);
					setApiError("");
				}
			} catch {
				if (active)
					setApiError(
						"Local API unavailable. Start it with uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000.",
					);
			}
		};
		void refresh();
		const interval = window.setInterval(refresh, 15000);
		return () => {
			active = false;
			window.clearInterval(interval);
		};
	}, []);

	const navigate = (nextPage: PageId) => {
		location.hash = nextPage === "home" ? "" : nextPage;
		setPage(nextPage);
		window.scrollTo({ top: 0, behavior: "smooth" });
	};
	return { page, navigate, health, metadata, apiError };
}

function App() {
	const { page, navigate, health, metadata, apiError } = useLocalState();
	const selectedTask = tasks.find((task) => task.id === page);

	return (
		<div className="app-shell">
			<header className="topbar">
				<button
					className="brand"
					onClick={() => navigate("home")}
					aria-label="Everyday AI home"
				>
					<span className="brand-mark">
						<Cpu
							size={18}
							strokeWidth={2.2}
						/>
					</span>
					<span>
						EVERYDAY <b>AI</b>
					</span>
				</button>
				<nav
					className="topnav"
					aria-label="Main navigation"
				>
					<button
						className={
							page === "home" ||
							selectedTask
								? "active"
								: ""
						}
						onClick={() => navigate("home")}
					>
						Challenges
					</button>
					<button
						className={
							page === "results"
								? "active"
								: ""
						}
						onClick={() =>
							navigate("results")
						}
					>
						The fight
					</button>
					<button
						className={
							page === "sources"
								? "active"
								: ""
						}
						onClick={() =>
							navigate("sources")
						}
					>
						Field notes
					</button>
				</nav>
				<div
					className="api-pill"
					title={apiError || "API state"}
				>
					<span
						className={`status-dot ${apiError ? "offline" : health ? "online" : "pending"}`}
					/>
					{apiError
						? "API offline"
						: health
							? "Local system"
							: "Connecting"}
				</div>
			</header>

			{apiError && (
				<div className="api-banner">
					<CircleAlert size={16} />
					{apiError}
				</div>
			)}

			<main>
				{page === "home" && (
					<HomePage
						health={health}
						navigate={navigate}
					/>
				)}
				{selectedTask && (
					<TaskPage
						task={selectedTask}
						readiness={
							health?.tasks[
								selectedTask.id
							] ?? null
						}
						metadata={
							metadata.tasks[
								selectedTask.id
							]
						}
					/>
				)}
				{page === "results" && (
					<ResultsPage
						metadata={metadata}
						readiness={health}
					/>
				)}
				{page === "sources" && <SourcesPage />}
			</main>

			<footer className="footer">
				<span>TECH SOCIETY · LOCAL ML LAB</span>
				<span>
					Three prototypes. No perfect answers.
				</span>
				<button onClick={() => navigate("sources")}>
					Sources & limits{" "}
					<ArrowUpRight size={14} />
				</button>
			</footer>
		</div>
	);
}

function HomePage({
	health,
	navigate,
}: {
	health: Health | null;
	navigate: (page: PageId) => void;
}) {
	return (
		<>
			<section className="hero-band">
				<div className="hero-copy">
					<div className="eyebrow hero-eyebrow">
						<span className="eyebrow-line" />
						A HANDS-ON MACHINE-LEARNING LAB
					</div>
					<h1>
						Beat the
						<br />
						<em>AI Lead.</em>
					</h1>
					<p>
						Three practical challenges. Two
						baselines each. The scoreboard
						only moves when the models do.
					</p>
					<button
						className="button button-lime"
						onClick={() =>
							navigate("results")
						}
					>
						Enter the fight{" "}
						<ArrowRight size={17} />
					</button>
				</div>
				<div
					className="hero-art"
					aria-label="Three model challenges represented as a benchmark board"
				>
					<div className="orbit orbit-a" />
					<div className="orbit orbit-b" />
					<div className="hero-stamp">
						<span>YOU</span>
						<strong>VS</strong>
						<span>AI LEAD</span>
					</div>
					<div className="art-tile tile-cv">
						<Camera size={20} />
						<b>CV</b>
						<span>01</span>
					</div>
					<div className="art-tile tile-nlp">
						<span className="quote-mark">
							Aa
						</span>
						<b>NLP</b>
						<span>02</span>
					</div>
					<div className="art-tile tile-tab">
						<Gauge size={20} />
						<b>TAB</b>
						<span>03</span>
					</div>
					<div className="art-caption">
						REAL DATA
						<br />
						LOCAL MODELS
					</div>
					<div className="hero-asterisk">✳</div>
				</div>
				<div className="hero-footer">
					<span>01 — 03</span>
					<span>SCROLL TO EXPLORE</span>
					<ArrowDownRight size={16} />
				</div>
			</section>

			<section className="section-pad challenge-section">
				<div className="section-heading">
					<div>
						<div className="eyebrow">
							<span className="eyebrow-line" />
							THE CHALLENGES
						</div>
						<h2>
							Small tasks.
							<br />
							<em>
								Useful
								questions.
							</em>
						</h2>
					</div>
					<p>
						Each model is a prototype, not
						an authority. Try it, inspect
						the evidence, then decide what
						you would change.
					</p>
				</div>
				<div className="challenge-list">
					{tasks.map((task) => (
						<ChallengeRow
							key={task.id}
							task={task}
							readiness={
								health?.tasks[
									task.id
								] ?? null
							}
							onOpen={() =>
								navigate(
									task.id,
								)
							}
						/>
					))}
				</div>
			</section>

			<section className="manifesto">
				<div className="manifesto-mark">
					<Sparkles size={25} />
				</div>
				<div className="manifesto-main">
					<div className="eyebrow">
						THE POINT ISN'T THE SCORE
					</div>
					<h2>
						Learn how a model
						<br />
						gets <em>something wrong.</em>
					</h2>
				</div>
				<div className="manifesto-side">
					<p>
						Swap the baseline. Change the
						data. Bring a better idea. Every
						result shown here comes from a
						local run, not a mock
						leaderboard.
					</p>
					<button
						className="text-link"
						onClick={() =>
							navigate("sources")
						}
					>
						Read the field notes{" "}
						<ArrowRight size={15} />
					</button>
				</div>
			</section>
		</>
	);
}

function ChallengeRow({
	task,
	readiness,
	onOpen,
}: {
	task: (typeof tasks)[number];
	readiness: TaskReadiness | null;
	onOpen: () => void;
}) {
	const state = readiness?.available_models.length
		? "Model ready"
		: "Model not trained";
	return (
		<button className="challenge-row" onClick={onOpen}>
			<span className="row-number">{task.number}</span>
			<span className="row-main">
				<span className="eyebrow">{task.eyebrow}</span>
				<strong>{task.name}</strong>
				<span>{task.summary}</span>
			</span>
			<span className="row-metric">
				<b>{state}</b>
				<span>{task.metric}</span>
			</span>
			<span className="row-arrow">
				<ArrowUpRight size={20} />
			</span>
		</button>
	);
}

function TaskPage({
	task,
	readiness,
	metadata,
}: {
	task: (typeof tasks)[number];
	readiness: TaskReadiness | null;
	metadata: TaskMetadata;
}) {
	return (
		<section className="task-page section-pad">
			<div className="task-kicker">
				<a href="#home">Challenges</a>
				<span>/</span>
				<span>{task.number}</span>
				<span className="eyebrow">{task.eyebrow}</span>
			</div>
			<div className="task-heading">
				<div>
					<div className="eyebrow">
						<span className="eyebrow-line" />
						CHALLENGE {task.number}
					</div>
					<h1>
						{task.name}
						<br />
						<em>
							{task.id === "vision"
								? "assistant."
								: task.id ===
									  "text"
									? "for review."
									: "estimator."}
						</em>
					</h1>
					<p>{task.summary}</p>
				</div>
				<ReadinessStamp readiness={readiness} />
			</div>
			<div className="task-workspace">
				<div className="task-interaction">
					{task.id === "vision" && (
						<VisionDemo
							readiness={readiness}
						/>
					)}
					{task.id === "text" && (
						<TextDemo
							readiness={readiness}
						/>
					)}
					{task.id === "tabular" && (
						<BikeDemo
							readiness={readiness}
						/>
					)}
				</div>
				<aside className="task-rail">
					<div className="rail-block">
						<div className="eyebrow">
							THE MODEL PAIR
						</div>
						<ModelPair
							metadata={metadata}
						/>
					</div>
					<div className="rail-block">
						<div className="eyebrow">
							WHAT IT DOESN'T DO
						</div>
						<p>
							{task.id === "vision"
								? "Does not identify an asset, assess damage, infer ownership, or update inventory. An unfamiliar item can still receive a known label."
								: task.id ===
									  "text"
									? "Not a phishing detector. No blocking or deleting. The dataset may not represent modern messages."
									: "Not a forecast or causal model. Historical observed weather is not future weather, and unusual input combinations may be out of distribution."}
						</p>
					</div>
					<div className="rail-block rail-experiment">
						<div className="eyebrow">
							<FlaskConical
								size={13}
							/>{" "}
							HOW TO BEAT THIS MODEL
						</div>
						<p>
							{task.id === "vision"
								? "Collect more distinct objects, then vary camera, lighting, and background. Keep objects separated across splits."
								: task.id ===
									  "text"
									? "Try character n-grams, threshold selection on validation data, or inspect false positives without exposing source messages."
									: "Compare chronological windows, add a documented seasonal feature, or study error by hour without leaking the target."}
						</p>
					</div>
					<BenchmarkMetrics
						taskId={task.id}
						metadata={metadata}
					/>
					{task.id === "vision" && (
						<VisionMistakes
							metadata={metadata}
						/>
					)}
				</aside>
			</div>
		</section>
	);
}

function ReadinessStamp({ readiness }: { readiness: TaskReadiness | null }) {
	const ready = Boolean(readiness?.available_models.length);
	return (
		<div
			className={`readiness-stamp ${ready ? "ready" : "waiting"}`}
		>
			<span className="status-dot" />
			{readiness
				? ready
					? `${readiness.available_models.length} MODEL${readiness.available_models.length > 1 ? "S" : ""} READY`
					: "MODEL NOT TRAINED"
				: "CHECKING LOCAL MODELS"}
		</div>
	);
}

function MissingModel({ readiness }: { readiness: TaskReadiness | null }) {
	return (
		<div className="missing-state">
			<div className="missing-icon">
				<Database size={22} />
			</div>
			<div className="eyebrow">
				{readiness?.status === "awaiting_data"
					? "AWAITING DATASET"
					: "LOCAL ARTIFACT NOT FOUND"}
			</div>
			<h3>Model not trained</h3>
			<p>
				{readiness?.status === "awaiting_data"
					? "The data is not present locally yet. No substitute data or prediction is being used."
					: "This demo only returns real local inference. Prepare the data and train a model to activate it."}
			</p>
			<code>
				{readiness?.next_command ??
					"Start the local API to check readiness."}
			</code>
		</div>
	);
}

function ModelPair({ metadata }: { metadata: TaskMetadata }) {
	const available = new Set(
		metadata.available_models.map((model) => model.model_id),
	);
	return (
		<div className="model-pair">
			{(["starter", "ai-lead"] as const).map((id) => (
				<div key={id} className="model-pair-row">
					<span className={`model-mini ${id}`}>
						{id === "starter" ? "S" : "AI"}
					</span>
					<span>
						<b>
							{id === "starter"
								? "Starter baseline"
								: "AI Lead"}
						</b>
						<small>
							{available.has(id)
								? "Artifact available"
								: "Not trained"}
						</small>
					</span>
					<span
						className={`model-dot ${available.has(id) ? "available" : ""}`}
					/>
				</div>
			))}
		</div>
	);
}

function BenchmarkMetrics({
	taskId,
	metadata,
}: {
	taskId: TaskId;
	metadata: TaskMetadata;
}) {
	const metrics = metadata.validation_metrics;
	const scoreName = taskId === "tabular" ? "mae" : "macro_f1";
	const scoreLabel =
		taskId === "tabular" ? "Validation MAE" : "Validation macro-F1";
	return (
		<div className="rail-block metric-rail">
			<div className="eyebrow">
				<Gauge size={13} />
				VALIDATION ONLY
			</div>
			{Object.keys(metrics).length ? (
				<div className="metric-lines">
					{(["starter", "ai-lead"] as const).map(
						(id) => {
							const value = Number(
								metrics[id]
									?.metrics?.[
									scoreName
								],
							);
							return (
								<div key={id}>
									<span>
										{id ===
										"starter"
											? "Starter"
											: "AI Lead"}
									</span>
									<b>
										{Number.isFinite(
											value,
										)
											? taskId ===
												"tabular"
												? `${value.toFixed(1)} rentals`
												: value.toFixed(
														3,
													)
											: "—"}
									</b>
								</div>
							);
						},
					)}
					<small>
						{scoreLabel}. Lower is better
						for MAE; higher for macro-F1.
					</small>
				</div>
			) : (
				<p>
					No measured validation results are
					available yet.
				</p>
			)}
		</div>
	);
}

function VisionMistakes({ metadata }: { metadata: TaskMetadata }) {
	const report =
		metadata.validation_metrics["ai-lead"] ??
		metadata.validation_metrics.starter;
	const mistakes = report?.representative_mistakes ?? [];
	if (!report || !mistakes.length)
		return (
			<div className="rail-block">
				<div className="eyebrow">
					REPRESENTATIVE VALIDATION MISTAKES
				</div>
				<p>
					Approved example mistakes will appear
					after the image dataset is trained and
					evaluated.
				</p>
			</div>
		);
	return (
		<div className="rail-block vision-mistakes">
			<div className="eyebrow">
				REPRESENTATIVE VALIDATION MISTAKES
			</div>
			{mistakes.map((mistake) => (
				<figure key={mistake.row_index}>
					<img
						src={apiUrl(
							`/api/vision/validation-examples/${mistake.row_index}/image?model_id=${report.model_id}`,
						)}
						alt={`Approved validation image. The model predicted ${mistake.predicted}; the recorded category is ${mistake.expected}.`}
					/>
					<figcaption>
						Predicted {mistake.predicted};
						expected {mistake.expected}.
					</figcaption>
				</figure>
			))}
		</div>
	);
}

function VisionDemo({ readiness }: { readiness: TaskReadiness | null }) {
	const [result, setResult] = useState<Prediction | null>(null);
	const [error, setError] = useState("");
	const [busy, setBusy] = useState(false);
	const [cameraOpen, setCameraOpen] = useState(false);
	const [model, setModel] = useState<ModelId>("ai-lead");
	const videoRef = useRef<HTMLVideoElement>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const ready = Boolean(readiness?.available_models.length);

	useEffect(
		() => () =>
			streamRef.current
				?.getTracks()
				.forEach((track) => track.stop()),
		[],
	);
	useEffect(() => {
		const available = readiness?.available_models ?? [];
		if (!available.includes(model))
			setModel(
				available.includes("ai-lead")
					? "ai-lead"
					: "starter",
			);
	}, [readiness, model]);

	const classify = async (file: File) => {
		if (!ready) return;
		setBusy(true);
		setError("");
		setResult(null);
		try {
			const form = new FormData();
			form.append("file", file);
			form.append("model_id", model);
			const response = await fetch(
				apiUrl("/api/predict/vision"),
				{
					method: "POST",
					body: form,
				},
			);
			const body = await response.json();
			if (!response.ok)
				throw new Error(
					body.detail?.status ===
						"model_not_trained"
						? "Model not trained. Prepare the image dataset and train the selected model."
						: String(
								body.detail ??
									"Image could not be classified.",
							),
				);
			setResult(body);
		} catch (reason) {
			setError(
				reason instanceof Error
					? reason.message
					: "Unable to reach the local API.",
			);
		} finally {
			setBusy(false);
		}
	};

	const startCamera = async () => {
		setError("");
		try {
			const stream =
				await navigator.mediaDevices.getUserMedia({
					video: { facingMode: "environment" },
					audio: false,
				});
			streamRef.current = stream;
			setCameraOpen(true);
			requestAnimationFrame(() => {
				if (videoRef.current)
					videoRef.current.srcObject = stream;
			});
		} catch {
			setError(
				"Camera permission was unavailable. Upload a local image instead.",
			);
		}
	};

	const capture = () => {
		const video = videoRef.current;
		if (!video || !video.videoWidth) return;
		const canvas = document.createElement("canvas");
		canvas.width = video.videoWidth;
		canvas.height = video.videoHeight;
		canvas.getContext("2d")?.drawImage(video, 0, 0);
		canvas.toBlob(
			(blob) => {
				if (blob)
					void classify(
						new File(
							[blob],
							"camera-frame.jpg",
							{ type: "image/jpeg" },
						),
					);
			},
			"image/jpeg",
			0.9,
		);
	};
	const closeCamera = () => {
		streamRef.current?.getTracks().forEach((track) => track.stop());
		streamRef.current = null;
		setCameraOpen(false);
	};

	if (!ready) return <MissingModel readiness={readiness} />;
	return (
		<>
			<div className="demo-title">
				<div className="eyebrow">
					ONE OBJECT · THREE LABELS
				</div>
				<h2>What did you bring?</h2>
				<p>
					One photographed equipment item. A
					person confirms the category.
				</p>
			</div>
			<label className="vision-model-select">
				<span className="field-label">Model</span>
				<select
					value={model}
					onChange={(event) =>
						setModel(
							event.target
								.value as ModelId,
						)
					}
				>
					{(["starter", "ai-lead"] as const)
						.filter((id) =>
							readiness?.available_models.includes(
								id,
							),
						)
						.map((id) => (
							<option
								key={id}
								value={id}
							>
								{id ===
								"starter"
									? "Starter · frozen ResNet-18"
									: "AI Lead · fine-tuned ResNet-18"}
							</option>
						))}
				</select>
			</label>
			{cameraOpen ? (
				<div className="camera-box">
					<video
						ref={videoRef}
						autoPlay
						playsInline
					/>
					<div className="camera-actions">
						<button
							className="button button-lime"
							onClick={capture}
						>
							<Camera size={16} />{" "}
							Capture frame
						</button>
						<button
							className="icon-button"
							onClick={closeCamera}
							title="Close camera"
						>
							<X size={17} />
						</button>
					</div>
				</div>
			) : (
				<div className="upload-zone">
					<FileImage size={26} />
					<h3>Choose a photo</h3>
					<p>
						JPEG, PNG or WebP · up to 8 MiB
						· processed in memory only
					</p>
					<div className="upload-actions">
						<label className="button button-dark">
							<Upload size={16} />{" "}
							Upload image
							<input
								type="file"
								accept="image/jpeg,image/png,image/webp"
								onChange={(
									event,
								) => {
									const file =
										event
											.target
											.files?.[0];
									if (
										file
									)
										void classify(
											file,
										);
									event.currentTarget.value =
										"";
								}}
							/>
						</label>
						<button
							className="button button-outline"
							onClick={startCamera}
						>
							<Camera size={16} /> Use
							camera
						</button>
					</div>
				</div>
			)}
			{busy && (
				<div className="inline-status">
					<LoaderCircle
						className="spin"
						size={17}
					/>{" "}
					Running local inference…
				</div>
			)}
			{error && (
				<div className="form-error">
					<CircleAlert size={16} />
					{error}
				</div>
			)}
			{result && <PredictionResult result={result} />}
			<div className="safety-note">
				<ShieldAlert size={16} />
				<span>
					Suggestion only. Scores are not
					guarantees; unfamiliar objects still
					receive a known-class prediction.
				</span>
			</div>
		</>
	);
}

function TextDemo({ readiness }: { readiness: TaskReadiness | null }) {
	const ready = Boolean(readiness?.available_models.length);
	const [text, setText] = useState("");
	const [model, setModel] = useState<ModelId>("ai-lead");
	const [result, setResult] = useState<Prediction | null>(null);
	const [error, setError] = useState("");
	const [busy, setBusy] = useState(false);
	if (!ready) return <MissingModel readiness={readiness} />;

	const submit = async (event: React.FormEvent) => {
		event.preventDefault();
		setBusy(true);
		setError("");
		setResult(null);
		try {
			const response = await fetch(
				apiUrl("/api/predict/text"),
				{
					method: "POST",
					headers: {
						"Content-Type":
							"application/json",
					},
					body: JSON.stringify({
						text,
						model_id: model,
					}),
				},
			);
			const body = await response.json();
			if (!response.ok)
				throw new Error(
					body.detail?.status ===
						"model_not_trained"
						? "Selected model is not trained."
						: String(
								body.detail ??
									"Prediction failed.",
							),
				);
			setResult(body);
		} catch (reason) {
			setError(
				reason instanceof Error
					? reason.message
					: "Unable to reach the local API.",
			);
		} finally {
			setBusy(false);
		}
	};
	return (
		<>
			<div className="demo-title">
				<div className="eyebrow">
					HUMAN REVIEW · NO MESSAGE LOGGING
				</div>
				<h2>Does this resemble spam?</h2>
				<p>
					Only text you enter is analyzed. It is
					not stored by the demo.
				</p>
			</div>
			<form className="text-form" onSubmit={submit}>
				<label
					className="field-label"
					htmlFor="message"
				>
					Message text
				</label>
				<textarea
					id="message"
					maxLength={5000}
					value={text}
					onChange={(event) =>
						setText(event.target.value)
					}
					placeholder="Paste a message to inspect…"
					required
				/>
				<div className="textarea-foot">
					<span>{text.length} / 5000</span>
					<span>
						Local inference · request body
						not logged
					</span>
				</div>
				<div className="fictional-examples">
					<span className="eyebrow">
						FICTIONAL EXAMPLES
					</span>
					{fictionalMessages.map((example) => (
						<button
							type="button"
							key={example.label}
							onClick={() => {
								setText(
									example.text,
								);
								setResult(null);
							}}
						>
							{example.label}
						</button>
					))}
				</div>
				<div className="form-controls">
					<label
						className="field-label"
						htmlFor="text-model"
					>
						Model
					</label>
					<select
						id="text-model"
						value={model}
						onChange={(event) =>
							setModel(
								event.target
									.value as ModelId,
							)
						}
					>
						{(
							[
								"starter",
								"ai-lead",
							] as const
						).map((id) => (
							<option
								value={id}
								key={id}
							>
								{id ===
								"starter"
									? "Starter · Naive Bayes"
									: "AI Lead · Logistic regression"}
							</option>
						))}
					</select>
					<button
						className="button button-dark"
						disabled={!text.trim() || busy}
					>
						{busy ? (
							<LoaderCircle
								className="spin"
								size={16}
							/>
						) : (
							<Play size={15} />
						)}{" "}
						Classify
					</button>
				</div>
			</form>
			{error && (
				<div className="form-error">
					<CircleAlert size={16} />
					{error}
				</div>
			)}
			{busy && (
				<div className="inline-status">
					<LoaderCircle
						className="spin"
						size={17}
					/>{" "}
					Running local inference…
				</div>
			)}
			{result && <PredictionResult result={result} />}
			<div className="safety-note">
				<ShieldAlert size={16} />
				<span>
					Not a phishing detector. Scores are not
					calibrated probabilities; do not use
					them to block or delete messages.
				</span>
			</div>
		</>
	);
}

const initialBike = {
	season: 2,
	year: 1,
	month: 6,
	hour: 12,
	holiday: 0,
	weekday: 2,
	workingday: 1,
	weathersit: 1,
	temperature_c: 20.5,
	feels_like_c: 25,
	humidity_percent: 60,
	wind_speed_kmh: 12,
};

function BikeDemo({ readiness }: { readiness: TaskReadiness | null }) {
	const ready = Boolean(readiness?.available_models.length);
	const [values, setValues] = useState(initialBike);
	const [result, setResult] = useState<Prediction | null>(null);
	const [error, setError] = useState("");
	const [busy, setBusy] = useState(false);
	if (!ready) return <MissingModel readiness={readiness} />;
	const update = (key: keyof typeof initialBike, value: number) =>
		setValues((current) => ({ ...current, [key]: value }));
	const submit = async (event: React.FormEvent) => {
		event.preventDefault();
		setBusy(true);
		setError("");
		setResult(null);
		try {
			const response = await fetch(
				apiUrl("/api/predict/tabular"),
				{
					method: "POST",
					headers: {
						"Content-Type":
							"application/json",
					},
					body: JSON.stringify(values),
				},
			);
			const body = await response.json();
			if (!response.ok)
				throw new Error(
					body.detail?.status ===
						"model_not_trained"
						? "No bike-demand artifact is trained."
						: String(
								body.detail ??
									"Estimation failed.",
							),
				);
			setResult(body);
		} catch (reason) {
			setError(
				reason instanceof Error
					? reason.message
					: "Unable to reach the local API.",
			);
		} finally {
			setBusy(false);
		}
	};
	return (
		<>
			<div className="demo-title">
				<div className="eyebrow">
					HISTORICAL SCENARIO · HOURLY TOTAL
				</div>
				<h2>Set the conditions.</h2>
				<p>
					Estimate total rentals for one supplied
					calendar and weather scenario.
				</p>
			</div>
			<form className="bike-form" onSubmit={submit}>
				<div className="input-grid">
					<NumberControl
						label="Hour"
						value={values.hour}
						min={0}
						max={23}
						suffix=":00"
						onChange={(value) =>
							update("hour", value)
						}
					/>
					<SelectControl
						label="Month"
						value={values.month}
						options={Array.from(
							{ length: 12 },
							(_, index) =>
								[
									index +
										1,
									new Date(
										2000,
										index,
									).toLocaleString(
										"en",
										{
											month: "long",
										},
									),
								] as [
									number,
									string,
								],
						)}
						onChange={(value) =>
							update("month", value)
						}
					/>
					<SelectControl
						label="Dataset year"
						value={values.year}
						options={[
							[0, "2011"],
							[1, "2012"],
						]}
						onChange={(value) =>
							update("year", value)
						}
					/>
					<SelectControl
						label="Weekday"
						value={values.weekday}
						options={[
							[0, "Sunday"],
							[1, "Monday"],
							[2, "Tuesday"],
							[3, "Wednesday"],
							[4, "Thursday"],
							[5, "Friday"],
							[6, "Saturday"],
						]}
						onChange={(value) =>
							update("weekday", value)
						}
					/>
					<SelectControl
						label="Season"
						value={values.season}
						options={[
							[1, "Dataset season 1"],
							[2, "Dataset season 2"],
							[3, "Dataset season 3"],
							[4, "Dataset season 4"],
						]}
						onChange={(value) =>
							update("season", value)
						}
					/>
					<SelectControl
						label="Weather"
						value={values.weathersit}
						options={[
							[
								1,
								"Clear / partly cloudy",
							],
							[2, "Mist / cloudy"],
							[
								3,
								"Light rain / snow",
							],
							[
								4,
								"Heavy rain / snow",
							],
						]}
						onChange={(value) =>
							update(
								"weathersit",
								value,
							)
						}
					/>
					<NumberControl
						label="Temperature"
						value={values.temperature_c}
						min={0}
						max={41}
						step={0.5}
						suffix="°C"
						onChange={(value) =>
							update(
								"temperature_c",
								value,
							)
						}
					/>
					<NumberControl
						label="Feels like"
						value={values.feels_like_c}
						min={0}
						max={50}
						step={0.5}
						suffix="°C"
						onChange={(value) =>
							update(
								"feels_like_c",
								value,
							)
						}
					/>
					<NumberControl
						label="Humidity"
						value={values.humidity_percent}
						min={0}
						max={100}
						suffix="%"
						onChange={(value) =>
							update(
								"humidity_percent",
								value,
							)
						}
					/>
					<NumberControl
						label="Wind speed"
						value={values.wind_speed_kmh}
						min={0}
						max={67}
						step={0.5}
						suffix="km/h"
						onChange={(value) =>
							update(
								"wind_speed_kmh",
								value,
							)
						}
					/>
				</div>
				<div className="toggle-row">
					<Toggle
						label="Working day"
						checked={Boolean(
							values.workingday,
						)}
						onChange={(checked) =>
							update(
								"workingday",
								checked ? 1 : 0,
							)
						}
					/>
					<Toggle
						label="Holiday"
						checked={Boolean(
							values.holiday,
						)}
						onChange={(checked) =>
							update(
								"holiday",
								checked ? 1 : 0,
							)
						}
					/>
				</div>
				<button
					className="button button-dark"
					disabled={busy}
				>
					{busy ? (
						<LoaderCircle
							className="spin"
							size={16}
						/>
					) : (
						<Gauge size={16} />
					)}{" "}
					Estimate demand
				</button>
			</form>
			{error && (
				<div className="form-error">
					<CircleAlert size={16} />
					{error}
				</div>
			)}
			{busy && (
				<div className="inline-status">
					<LoaderCircle
						className="spin"
						size={17}
					/>{" "}
					Estimating locally…
				</div>
			)}
			{result && <BikeResult result={result} />}
			<div className="safety-note">
				<Info size={16} />
				<span>
					Observed weather is not a forecast.
					Arbitrary scenarios may be outside the
					training data; no prediction interval is
					provided.
				</span>
			</div>
		</>
	);
}

function NumberControl({
	label,
	value,
	min,
	max,
	step = 1,
	suffix,
	onChange,
}: {
	label: string;
	value: number;
	min: number;
	max: number;
	step?: number;
	suffix?: string;
	onChange: (value: number) => void;
}) {
	return (
		<label className="number-control">
			<span>{label}</span>
			<span className="number-input-wrap">
				<input
					type="number"
					value={value}
					min={min}
					max={max}
					step={step}
					onChange={(event) =>
						onChange(
							Number(
								event.target
									.value,
							),
						)
					}
				/>
				<small>{suffix}</small>
			</span>
		</label>
	);
}

function SelectControl({
	label,
	value,
	options,
	onChange,
}: {
	label: string;
	value: number;
	options: [number, string][];
	onChange: (value: number) => void;
}) {
	return (
		<label className="number-control">
			<span>{label}</span>
			<span className="select-wrap">
				<select
					value={value}
					onChange={(event) =>
						onChange(
							Number(
								event.target
									.value,
							),
						)
					}
				>
					{options.map(([optionValue, text]) => (
						<option
							key={optionValue}
							value={optionValue}
						>
							{text}
						</option>
					))}
				</select>
				<ChevronDown size={14} />
			</span>
		</label>
	);
}

function Toggle({
	label,
	checked,
	onChange,
}: {
	label: string;
	checked: boolean;
	onChange: (checked: boolean) => void;
}) {
	return (
		<label className="switch-control">
			<input
				type="checkbox"
				checked={checked}
				onChange={(event) =>
					onChange(event.target.checked)
				}
			/>
			<span className="switch-track" />
			<span>{label}</span>
		</label>
	);
}

function PredictionResult({ result }: { result: Prediction }) {
	const scores = (result.scores ?? {}) as Record<string, number>;
	return (
		<section className="prediction-result" aria-live="polite">
			<div className="result-top">
				<div>
					<div className="eyebrow">
						LOCAL MODEL SUGGESTION
					</div>
					<h3>{String(result.prediction)}</h3>
				</div>
				<div className="result-time">
					<Clock3 size={14} />
					{result.inference_ms
						? `${String(result.inference_ms)} ms`
						: String(result.model_id)}
				</div>
			</div>
			{!!Object.keys(scores).length && (
				<div className="score-list">
					{Object.entries(scores)
						.sort((a, b) => b[1] - a[1])
						.map(([label, score]) => (
							<div
								className="score-row"
								key={label}
							>
								<span>
									{label}
								</span>
								<div className="score-track">
									<i
										style={{
											width: `${Math.min(100, Math.max(0, score * 100))}%`,
										}}
									/>
								</div>
								<b>
									{(
										score *
										100
									).toFixed(
										1,
									)}
									%
								</b>
							</div>
						))}
				</div>
			)}
			{typeof result.score_label === "string" && (
				<p className="score-semantics">
					{String(result.score_label)}
				</p>
			)}
			{typeof result.caveat === "string" && (
				<p>{result.caveat}</p>
			)}
			<small>
				Model version:{" "}
				{String(
					result.model_version ??
						result.model_id ??
						"local artifact",
				)}
			</small>
		</section>
	);
}

function BikeResult({ result }: { result: Prediction }) {
	const predictions = (result.predictions ?? {}) as Record<
		string,
		number
	>;
	return (
		<section
			className="prediction-result bike-result"
			aria-live="polite"
		>
			<div className="result-top">
				<div>
					<div className="eyebrow">
						HISTORICAL SCENARIO ESTIMATE
					</div>
					<h3>Rental demand</h3>
				</div>
				<span className="result-time">
					rentals / hour
				</span>
			</div>
			<div className="bike-estimates">
				{Object.entries(predictions).map(
					([id, value]) => (
						<div key={id}>
							<small>
								{id ===
								"starter"
									? "Starter baseline"
									: "AI Lead"}
							</small>
							<strong>
								{Math.round(
									value,
								).toLocaleString()}
							</strong>
							<span>rentals</span>
						</div>
					),
				)}
			</div>
			<p>
				{(
					result.warnings as string[] | undefined
				)?.join(" ")}
			</p>
		</section>
	);
}

function PredictionSubmission({
	taskId,
	enabled,
	onComplete,
}: {
	taskId: TaskId;
	enabled: boolean;
	onComplete: (metrics: Record<string, unknown>) => void;
}) {
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");
	const submit = async (file: File) => {
		setBusy(true);
		setError("");
		try {
			const form = new FormData();
			form.append("file", file);
			const response = await fetch(
				apiUrl(`/api/submissions/${taskId}`),
				{
					method: "POST",
					body: form,
				},
			);
			const body = await response.json();
			if (!response.ok)
				throw new Error(
					String(
						body.detail ??
							"Prediction CSV was rejected.",
					),
				);
			onComplete(body.metrics as Record<string, unknown>);
		} catch (reason) {
			setError(
				reason instanceof Error
					? reason.message
					: "Unable to reach the local API.",
			);
		} finally {
			setBusy(false);
		}
	};
	return (
		<div className="prediction-submission">
			<div className="eyebrow">
				YOU · VALIDATION PREDICTIONS
			</div>
			{enabled ? (
				<div className="prediction-submission-actions">
					<a
						className="button button-outline"
						href={apiUrl(
							`/api/submissions/${taskId}/template`,
						)}
						download
					>
						<Download size={15} /> Get row
						IDs
					</a>
					<label
						className={`button button-lime ${busy ? "disabled" : ""}`}
					>
						{busy ? (
							<LoaderCircle
								className="spin"
								size={15}
							/>
						) : (
							<Upload size={15} />
						)}
						Upload CSV
						<input
							className="csv-upload-input"
							type="file"
							accept=".csv,text/csv"
							disabled={busy}
							onChange={(event) => {
								const file =
									event
										.currentTarget
										.files?.[0];
								if (file)
									void submit(
										file,
									);
								event.currentTarget.value =
									"";
							}}
						/>
					</label>
				</div>
			) : (
				<p className="submission-unavailable">
					Prediction upload unavailable until a
					local model and validation set exist.
				</p>
			)}
			<p className="submission-safety">
				CSV predictions only. Model code is never
				accepted; scoring uses validation rows only.
			</p>
			{error && (
				<p className="submission-error" role="alert">
					{error}
				</p>
			)}
		</div>
	);
}

function ResultsPage({
	metadata,
	readiness,
}: {
	metadata: Metadata;
	readiness: Health | null;
}) {
	const [taskId, setTaskId] = useState<TaskId>("text");
	const [slots, setSlots] = useState<{
		you: FighterId | null;
	}>({ you: null });
	const [submissionMetrics, setSubmissionMetrics] = useState<Record<
		string,
		unknown
	> | null>(null);
	const taskMeta = metadata.tasks[taskId];
	const onDrop = (event: React.DragEvent) => {
		event.preventDefault();
		const value = event.dataTransfer.getData(
			"text/model",
		) as ModelId;
		if (value === "starter") setSlots({ you: value });
	};
	const modelMetrics = (model: FighterId | null) =>
		model === "student"
			? submissionMetrics
			: model
				? taskMeta.validation_metrics[model]?.metrics
				: null;
	const opponentReady =
		readiness?.tasks[taskId].available_models.includes("ai-lead") ??
		false;
	const opponentMetrics = modelMetrics("ai-lead");
	const score = (model: FighterId | null) => {
		const metric = modelMetrics(model);
		return metric
			? taskId === "tabular"
				? `${Number(metric.mae).toFixed(1)} rentals MAE`
				: `${Number(metric.macro_f1).toFixed(3)} macro-F1`
			: "No measured result";
	};
	return (
		<section className="section-pad results-page">
			<div className="eyebrow">
				<span className="eyebrow-line" />
				THE BENCHMARK FLOOR
			</div>
			<div className="results-heading">
				<h1>
					YOU <em>vs</em>
					<br />
					AI LEAD.
				</h1>
				<p>
					Drag a locally trained baseline into the
					ring. Results come from validation data
					only; the stronger model is not assumed
					to win.
				</p>
			</div>
			<div className="benchmark-toolbar">
				<div className="eyebrow">
					CHOOSE A CHALLENGE
				</div>
				<div className="segmented-control">
					{tasks.map((task) => (
						<button
							key={task.id}
							className={
								taskId ===
								task.id
									? "selected"
									: ""
							}
							onClick={() => {
								setTaskId(
									task.id,
								);
								setSlots({
									you: null,
								});
								setSubmissionMetrics(
									null,
								);
							}}
						>
							{task.symbol}
						</button>
					))}
				</div>
			</div>
			<div className="fight-board">
				<div className="fight-sources">
					<div className="eyebrow">
						AVAILABLE BASELINES
					</div>
					{(["starter"] as const).map((id) => {
						const available =
							readiness?.tasks[
								taskId
							].available_models.includes(
								id,
							) ?? false;
						return (
							<div
								key={id}
								className={`draggable-model ${available ? "can-drag" : "unavailable"}`}
								draggable={
									available
								}
								onDragStart={(
									event,
								) => {
									event.dataTransfer.setData(
										"text/model",
										id,
									);
								}}
							>
								<span
									className={`model-mini ${id}`}
								>
									{id ===
									"starter"
										? "S"
										: "AI"}
								</span>
								<div>
									<b>
										{id ===
										"starter"
											? "Starter baseline"
											: "AI Lead"}
									</b>
									<small>
										{available
											? "Drag into the fight"
											: "Model not trained"}
									</small>
								</div>
								<ArrowUpRight
									size={
										15
									}
								/>
							</div>
						);
					})}
					<PredictionSubmission
						taskId={taskId}
						enabled={Boolean(
							readiness?.tasks[
								taskId
							].available_models.includes(
								"starter",
							) &&
							readiness?.tasks[
								taskId
							].available_models.includes(
								"ai-lead",
							),
						)}
						onComplete={(metrics) => {
							setSubmissionMetrics(
								metrics,
							);
							setSlots({
								you: "student",
							});
						}}
					/>
				</div>
				<div className="fight-ring">
					<div className="ring-topline">
						<span>VALIDATION ARENA</span>
						<span>
							<LockKeyhole
								size={12}
							/>{" "}
							FINAL TEST SEALED
						</span>
					</div>
					<div className="fighter-slots">
						<div
							className="fighter-slot you"
							onDragOver={(event) =>
								event.preventDefault()
							}
							onDrop={onDrop}
						>
							<span className="fighter-label">
								YOU
							</span>
							<div className="fighter-symbol">
								{slots.you ===
								"student"
									? "YOU"
									: slots.you ===
										  "starter"
										? "S"
										: "?"}
							</div>
							<b>
								{slots.you ===
								"student"
									? "Your predictions"
									: slots.you ===
										  "starter"
										? "Starter baseline"
										: "Drop Starter or upload CSV"}
							</b>
							<strong>
								{score(
									slots.you,
								)}
							</strong>
							{modelMetrics(
								slots.you,
							) && (
								<span className="metric-detail">
									{taskId ===
									"tabular"
										? `RMSE ${Number(modelMetrics(slots.you)?.rmse).toFixed(1)}`
										: `Accuracy ${Number(modelMetrics(slots.you)?.accuracy).toFixed(3)}`}
								</span>
							)}
						</div>
						<div className="fighter-slot lead fixed-opponent">
							<span className="fighter-label">
								PRELOADED
								OPPONENT
							</span>
							<div className="fighter-symbol">
								AI
							</div>
							<b>AI Lead</b>
							<strong>
								{opponentReady
									? score(
											"ai-lead",
										)
									: "Model not trained"}
							</strong>
							{opponentMetrics && (
								<span className="metric-detail">
									{taskId ===
									"tabular"
										? `RMSE ${Number(opponentMetrics.rmse).toFixed(1)}`
										: `Accuracy ${Number(opponentMetrics.accuracy).toFixed(3)}`}
								</span>
							)}
						</div>
					</div>
					<div className="fight-verdict">
						{slots.you &&
						opponentReady &&
						modelMetrics(slots.you) &&
						opponentMetrics ? (
							<>
								<Check
									size={
										15
									}
								/>{" "}
								Measured on the
								same validation
								split. This is
								not a
								private-test
								score.
							</>
						) : (
							<>
								<Info
									size={
										15
									}
								/>{" "}
								Choose available
								models with
								measured
								validation
								results to
								compare.
							</>
						)}
					</div>
				</div>
			</div>
			{taskId === "tabular" && (
				<BikeBenchmarkCharts
					reports={taskMeta.validation_metrics}
				/>
			)}
			<div className="experiment-strip">
				<FlaskConical size={17} />
				<div>
					<b>How to beat this model</b>
					<span>
						Change one thing at a time: data
						coverage, feature
						representation, split design, or
						a validation-selected decision
						threshold.
					</span>
				</div>
				<button
					onClick={() =>
						setTaskId(
							taskId === "vision"
								? "text"
								: taskId ===
									  "text"
									? "tabular"
									: "vision",
						)
					}
				>
					Next challenge <ArrowRight size={15} />
				</button>
			</div>
			<div className="metric-explainer">
				<div>
					<span>TEXT</span>
					<b>Spam precision · recall · F1</b>
					<small>
						False positives flag ordinary
						texts; false negatives miss
						spam.
					</small>
				</div>
				<div>
					<span>VISION</span>
					<b>Macro-F1 · accuracy</b>
					<small>
						Per-class results matter when
						class counts differ.
					</small>
				</div>
				<div>
					<span>TABULAR</span>
					<b>MAE · RMSE</b>
					<small>
						Typical error in hourly
						rental-count units.
					</small>
				</div>
			</div>
		</section>
	);
}

function BikeBenchmarkCharts({
	reports,
}: {
	reports: TaskMetadata["validation_metrics"];
}) {
	const report = reports["ai-lead"] ?? reports.starter;
	const metrics = report?.metrics;
	const samples =
		(metrics?.predicted_actual_sample as
			| { predicted: number; actual: number }[]
			| undefined) ?? [];
	const errors =
		(metrics?.errors_by_hour as
			| Record<string, number>
			| undefined) ?? {};
	if (!samples.length)
		return (
			<div className="chart-empty">
				No measured validation predictions are available
				for the chart.
			</div>
		);
	const maxValue = Math.max(
		1,
		...samples.flatMap((point) => [point.predicted, point.actual]),
	);
	const maxError = Math.max(1, ...Object.values(errors));
	const plot = { left: 38, top: 14, width: 400, height: 190 };
	return (
		<section className="bike-charts">
			<div className="chart-heading">
				<div>
					<div className="eyebrow">
						VALIDATION ·{" "}
						{report?.model_id === "ai-lead"
							? "AI LEAD"
							: "STARTER"}
					</div>
					<h2>Where demand estimates land</h2>
				</div>
				<span>rentals / hour</span>
			</div>
			<div className="scatter-wrap">
				<svg
					viewBox="0 0 480 242"
					role="img"
					aria-label="Predicted versus actual bike rentals with perfect prediction reference line"
				>
					<line
						x1={plot.left}
						y1={plot.top + plot.height}
						x2={plot.left + plot.width}
						y2={plot.top + plot.height}
						className="chart-axis"
					/>
					<line
						x1={plot.left}
						y1={plot.top}
						x2={plot.left}
						y2={plot.top + plot.height}
						className="chart-axis"
					/>
					<line
						x1={plot.left}
						y1={plot.top + plot.height}
						x2={plot.left + plot.width}
						y2={plot.top}
						className="chart-reference"
					/>
					{samples
						.slice(0, 250)
						.map((point, index) => (
							<circle
								key={index}
								cx={
									plot.left +
									(point.actual /
										maxValue) *
										plot.width
								}
								cy={
									plot.top +
									plot.height -
									(point.predicted /
										maxValue) *
										plot.height
								}
								r="2.5"
								className="chart-point"
							/>
						))}
					<text
						x="238"
						y="235"
						textAnchor="middle"
					>
						Actual
					</text>
					<text
						x="12"
						y="109"
						textAnchor="middle"
						transform="rotate(-90 12 109)"
					>
						Predicted
					</text>
					<text x="35" y="220">
						0
					</text>
					<text x="432" y="220">
						{Math.round(maxValue)}
					</text>
					<text x="18" y="18">
						{Math.round(maxValue)}
					</text>
				</svg>
				<div className="chart-legend">
					<span>
						<i className="legend-dot" />
						Validation rows
					</span>
					<span>
						<i className="legend-line" />
						Perfect prediction
					</span>
				</div>
			</div>
			<div className="hour-errors">
				<div className="eyebrow">
					MEAN ABSOLUTE ERROR BY HOUR
				</div>
				{Object.entries(errors)
					.sort(
						([left], [right]) =>
							Number(left) -
							Number(right),
					)
					.map(([hour, value]) => (
						<div
							className="hour-error-row"
							key={hour}
						>
							<span>
								{hour.padStart(
									2,
									"0",
								)}
								:00
							</span>
							<div>
								<i
									style={{
										width: `${Math.max(1, (value / maxError) * 100)}%`,
									}}
								/>
							</div>
							<b>
								{value.toFixed(
									0,
								)}
							</b>
						</div>
					))}
			</div>
		</section>
	);
}

function SourcesPage() {
	return (
		<section className="section-pad sources-page">
			<div className="eyebrow">
				<span className="eyebrow-line" />
				FIELD NOTES
			</div>
			<h1>
				Know what the
				<br />
				<em>model can’t know.</em>
			</h1>
			<div className="sources-grid">
				<article>
					<span className="source-number">
						01
					</span>
					<h2>Equipment images</h2>
					<p>
						Locally collected, consented
						pilot photos only. There is no
						bundled image dataset. One main
						object per image;
						object-disjoint splits prevent
						near-identical photos of one
						item appearing in both train and
						evaluation.
					</p>
					<a
						href="https://archive.ics.uci.edu/"
						target="_blank"
						rel="noreferrer"
					>
						Collection guide{" "}
						<ArrowUpRight size={14} />
					</a>
				</article>
				<article>
					<span className="source-number">
						02
					</span>
					<h2>SMS Spam Collection</h2>
					<p>
						UCI Machine Learning Repository,
						dataset 228. Attribution:
						Almeida, Gómez Hidalgo &
						Yamakami (2011). CC BY 4.0. Raw
						messages stay local and never
						appear in the UI.
					</p>
					<a
						href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
						target="_blank"
						rel="noreferrer"
					>
						UCI dataset page{" "}
						<ArrowUpRight size={14} />
					</a>
				</article>
				<article>
					<span className="source-number">
						03
					</span>
					<h2>Bike Sharing</h2>
					<p>
						UCI Machine Learning Repository,
						dataset 275. Hourly `cnt` is
						total rentals. `casual` and
						`registered` are excluded
						because they sum to the target;
						row IDs are excluded too. CC BY
						4.0.
					</p>
					<a
						href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
						target="_blank"
						rel="noreferrer"
					>
						UCI dataset page{" "}
						<ArrowUpRight size={14} />
					</a>
				</article>
			</div>
			<div className="limits-band">
				<div>
					<ShieldAlert size={19} />
					<b>Limits are part of the result.</b>
				</div>
				<p>
					These are educational prototypes, not
					production services. A category
					suggestion does not identify an asset.
					Spam scores are not a security
					guarantee. Bike demand estimates are
					historical scenarios, not forecasts or
					causal effects. No prediction intervals
					or fabricated performance are shown.
				</p>
			</div>
			<div className="privacy-band">
				<LockKeyhole size={18} />
				<div>
					<b>Local by design</b>
					<span>
						Inference runs on this machine.
						Uploaded images and typed
						messages are not saved by
						default. Local model artifacts
						are organiser-generated and must
						be treated as trusted files.
					</span>
				</div>
			</div>
		</section>
	);
}

export default App;
