# Changelog

All notable changes to MICRA (formerly Poly-X) are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Held-out test evaluation of torchvision Mask R-CNN (#243).** The held-out
  action now scores the torchvision family as well as Ultralytics detection and
  segmentation: a finished execution's frozen `checkpoints/best.pt` is rebuilt
  into its detector from `family.weights`, the cohort's class count and
  `protocol.image_size`, and its predictions go through the family's own
  inference path into the same `SharedEvaluator`. Both families write the same
  `heldout_test.json` shape and identifying fields, with the framework as the
  only difference. The export's heads carry `class_count + 1` rows, so they are
  resized before the state dict is loaded; a stock pretrained checkpoint is
  still transferred before its heads are replaced.
- **Held-out test evaluation (`polyx.pipeline.heldout`, #242).** A finished execution's
  frozen `checkpoints/best.pt` is scored once on the `test` fold through the
  shared evaluator, at one recorded confidence threshold, and the result is its
  own artifact (`heldout_test.json` + `heldout_test.md`) naming the source
  execution, the checkpoint SHA-256, the test-cohort fingerprint, the threshold,
  the overlap rule and the split. Predictions come from the frozen native
  weights, never from a new Lightning run. The action refuses an unfinished
  execution and any family without an inference path, never writes the
  execution's metrics, selection, training state or completion record, and
  reports `NOT EVALUATED` with a non-zero exit when the dataset declares no
  test split instead of substituting validation. Documented in the pipeline
  section of `CLAUDE.md`.
- **Article export (`polyx.gui.report.exportables`).** The report can now hand
  back what a manuscript needs instead of a screenshot of it: every table and
  figure is numbered in the document's language, and each one gets CSV + booktabs
  LaTeX (tables) or SVG + PDF + 300 dpi PNG (figures) next to the HTML, with
  download links inside the document and an `index.csv` tying each number to its
  file. Figure data is recovered from the drawing itself (bars, lines, scatter,
  heatmaps) so a histogram can be redrawn from its measurements rather than its
  bar heights. Off by default: `generate_report(..., article_export=True)`, and
  `export_folder=False` numbers the document without writing the folder. Ported
  from upstream (`polyx/core/exportables.py`, MIT, Cristofher Ferrada), with
  English identifiers, the gettext catalogs and one behavioural change: the
  per-particle measure tables in the cards section are not numbered as report
  tables.
- **Headless, model-agnostic training pipeline (`polyx-pipeline`).** A new
  parallel path beside the GUI Trainer composes `conf/` with Hydra and trains
  YOLO v8/v11 detection and segmentation and torchvision Mask R-CNN through one
  PyTorch Lightning loop under one framework-neutral protocol. It ships a shared
  COCO evaluator whose figures always name the criterion, best-weight selection
  by the with-class operating-point F1, resumable progress snapshots and real Lightning
  checkpoints, a COCO mask import/conversion path (machine masks are never
  ground truth), an Optuna sweeper multirun with a SQLite study on the work
  filesystem and a `sweep_best.json` artifact, hash-verified weights resolved
  before a framework loads them, and SLURM job scripts that stage weights and
  honour `POLYX_CONTAINER`. The GUI Trainer is unchanged. See `docs/hpc.md`,
  `docs/hpc-training-pipeline.md`, ADR-0007 to ADR-0010 and the
  `src/polyx/pipeline/` section in `CLAUDE.md`.
- Fast `Ruff` CI workflow (`ruff check` + `ruff format --check`) that installs
  the ruff version pinned in `uv.lock` and needs no project dependencies, so a
  lint failure reports in seconds instead of after the gate's suite.
- `pre-commit` CI workflow running every hook in `.pre-commit-config.yaml`, so
  a hook that was never installed locally still fails the PR.
- **Release checklist (`docs/RELEASE_CHECKLIST.md`).** The irreversible steps in
  order — version bump, changelog, tag on `main`, PyPI pre-release `2.1.0a1`,
  weight upload with `SHA256SUMS`, Zenodo archival, `CITATION.cff` with the DOI
  and `gh attestation verify` — with the pre-release rule stated once.

### Changed
- **Resume is bound to one execution and loads checkpoints restrictively (#241).**
  Every execution now writes `execution_identity.json` and embeds the same
  record in every Lightning checkpoint: resolved family (name, framework, task
  and weights declaration), initialization-weight identity (verified SHA-256
  digest when pinned, else the declared name), protocol and dataset
  fingerprint. A resume compares the requested execution against it and refuses
  any mismatch with an error naming the field. Lightning's absolute
  `best_model_path` is resolved inside the execution's own checkpoint
  directory, so a moved execution resumes and an external path is refused.
  Recovery metadata, epoch inspection, `fit` resume and re-validation all use
  restricted `weights_only=True` loading; there is no unrestricted fallback.
  **Behaviour change:** a checkpoint with no verifiable identity record — every
  checkpoint written before this version — is now refused with an explanation
  and left untouched. This supersedes the progress-record fallback #239 added
  for identity, protocol and dataset fingerprint: the heartbeat is never
  enough. Staged or imported weights initialise a new run and cannot be
  mistaken for a resume. `GLOSSARY.md` records **execution identity** and
  **resume checkpoint**.
- **Model selection is fixed to the validation split (#238).** A single
  training execution, a campaign job and a study trial refuse any
  `protocol.split` other than `val` before the first fit or trial — composed,
  overridden or searched — with an error that names `val` and says why: the
  held-out test cohort must never decide which checkpoint, family or trial is
  kept. The default is unchanged, a run that names no split still selects on
  `val`, and the study records that canonical value as its fixed split.
  `GLOSSARY.md` separates the **selection split** from the **held-out test
  split**, and the pipeline section of `CLAUDE.md` states the rule.
- **PyTorch moves to 2.13.0 and torchvision to 0.28.0 (#203).** The pair is
  raised in `pyproject.toml` and `uv.lock`, closing the five advisories
  Dependabot reports against 2.7.1/0.22.1, the last of which clears at 2.13.0.
  The cluster image keeps its CUDA 12.6 contract: the wheelhouse builder drops
  the lockfile's pinned CUDA tree by name — checked against `torch`'s closure
  in `uv.lock`, so a new `nvidia-*` package keeps its pin — and resolves the
  runtime from the cu126 wheels' own metadata, with the exported lockfile as
  constraints. The downloaded tree is then verified as `+cu126`/CUDA 12.6
  before the manifest is written. The upgrade does not make an untrusted
  checkpoint safe to load; the digest rule stands.
- **Pillow moves to 13.0.0 (#203).** The security pin is raised in
  `pyproject.toml` and `uv.lock`. The desktop ZIP
  channel that #203 introduced is retired by ADR-0011, so the fix reaches a
  checkout through `uv sync` and a cluster through the rebuilt image.
- **Weight downloads require the published digest.** `download()` takes
  `expected_sha256` as a required keyword argument, and `micra
  download-models` refuses without `--sha256` and `--url` before touching the
  filesystem or the network, naming the missing flag and saying why. No default
  URL ships until a Zenodo record or GitHub release exists, so the absence of
  weights is an explicit error instead of a 404. The HPC fetcher binds each
  manifest digest into the downloader, so the manifest stays the source of the
  hash.
- **The distribution is `micra` and the identity is truthful.** The wheel is
  named `micra`, `requires-python` is `>=3.11,<3.13` because the locked PySide6
  and numpy do not install on newer interpreters, the package URLs point at
  `Freston1605/micra`, and one `polyx.repository` constant feeds
  `download_models`. `[project].maintainers` records the maintainer without an
  address, the `LICENSE` keeps the original copyright and adds the
  modifications line, and `SECURITY.md` reports only through GitHub's private
  advisory form. The Detector credit and the HTML report footer name the
  original design, the maintainer and the laboratory. See ADR-0011.
- **The HPC pipeline's training lifecycle is Lightning's, not a second one.**
  `ModelCheckpoint` keeps the strictly best `with_class_score` and
  `EarlyStopping` measures patience against `min_delta`, so the two thresholds
  are the framework's own behaviour instead of a parallel implementation;
  callback state (patience and the kept best) travels inside the checkpoint and
  survives a preemption. Both families validate inside Lightning's validation
  loop, YOLO from the epoch's immutable native export because Ultralytics fuses
  the network it is handed. Checkpoints are written through an atomic
  `CheckpointIO`, and publication re-validates the kept checkpoint so
  `metrics.json` and `best.pt` always describe the same weights.
  `checkpoints/lightning/last.ckpt` is the resume point; the progress record is
  the heartbeat and the protocol/dataset guard.

### Removed
- **The consumer-distribution track (#212, ADR-0011).** `install/` (the
  installers, the auto-updater, the uninstaller, the migrator and the
  portable/macOS bundles), `requirements.txt`, the PyInstaller spec under
  `tools/packaging/`, `LEEME.txt` and `LEEME_macOS.md` are gone, along with the
  GUI update notice (`polyx.gui.updates`), `platform.name_updater()` and
  `platform.launch_updater()`. `empaquetar_dataset_para_drive.bat` moved to
  `tools/data/` because it packages a dataset, not the software. CI no longer
  smoke-tests the installers, and ruff no longer scans `install/`. The frozen
  manual stays on disk, out of scope and not regenerated, and every console
  script — the `polyx-*` aliases included — remains.

### Fixed
- **Upstream licence notice kept verbatim.** `LICENSE` now reads
  `2025-2026 Cristofher Ferrada` — the range upstream amended into
  `CrissFerrada/Poly-X-Microplastics@aaf25c0` — instead of a 2026-only line,
  with the maintainer's `(modifications)` line unchanged below it; MIT
  requires the original notice to be retained. `tests/test_identity.py` pins
  the range so a later rewrite cannot quietly narrow it again.
- **Fourth review of the HPC pipeline (#202).** Both container definitions now
  build: the project wheel is produced in the wheelhouse where `NOTICE` and the
  build backend exist, and the runtime installs it with `--no-index --no-deps`
  instead of building from source, with Apptainer selecting `/opt/venv` before
  its first install. `polyx-pipeline` no longer carries a second sweep loop:
  `python -m polyx.pipeline.sweep_app --multirun` is the only sweep entry point,
  and one allowlist refuses sweep fields outside `protocol.<searchable field>`
  and any direction other than maximize in both the written study record and
  the plugin's resolved distributions. A resume now compares a dataset
  fingerprint as well as the protocol, so relabelled images or an edited COCO
  export cannot keep the old checkpoint. The kept checkpoint is the strictly
  best score, so a gain smaller than `min_delta` still replaces `best.pt`.
  Mask R-CNN validation pastes masks in bounded chunks and stores them
  bit-packed, and runs one image per step, so a 4K split no longer asks the
  V100 for tens of gigabytes at once.
- **Third follow-up HPC review fixes.** A killed Optuna trial is recovered by
  its own id (resume its checkpoint, retry its parameters, or reconcile a score
  its finished execution already wrote), one trial at a time through a lease
  file; the winner export requires the record, completion marker, metrics and
  best checkpoint. A fresh run refuses a reused execution directory. The
  progress snapshot carries the fit's precision, device and verified weights so
  a recovered publication keeps them. Detection candidates are budgeted per
  class and the reference cap is applied per image and category; polygon masks
  use a port of pycocotools' rasterizer; Mask R-CNN retention follows the
  loader's ceil resize and segment resampling; collision checks use the label
  line the codec actually writes, and the conversion cache version covers the
  new rule.
- **Second follow-up HPC review fixes.** Interrupted Optuna trials are
  relaunched by their persisted id instead of being abandoned for a new trial.
  Pipeline failures carry invariant codes/parameters and translate only at
  display; completion is written after the final log line; a fresh run clears a
  stale marker; and the export requires the winner's metrics. Detection and
  evaluation now share a 300-prediction collection cap, masks are rasterized at
  COCO's 5x scale, segmentation uses the loader's segment-derived boxes and
  0.01 area threshold, colliding serialized polygons are refused, and the
  conversion cache key includes image bytes.
- **Follow-up HPC review fixes.** Checkpoint validation and `best.pt` now use
  the live weights without carrying a staged checkpoint's EMA; COCO AP has the
  reference per-image/category 100-detection cap while F1 applies the configured
  confidence threshold. Campaigns verify byte-identical image content and fold
  membership, use the pinned loader's own tiny-target rule, and refuse
  annotations the loader would collapse. Completion is recorded atomically
  after every artifact, sweeps pass the real Optuna trial id and persist the
  study identity for a strict winner lookup, and sync-back uses the child
  directory name. The OCI runtime creates the provenance directory, and
  container builds stamp the exact source SHA.
- **HPC pipeline review fixes.** YOLO validation now scores a fresh snapshot
  model instead of fusing and freezing the live training network, so training
  continues past the first epoch; GPU result tensors are converted to CPU
  before NumPy. Checkpoint selection and Optuna now reuse the project's
  with-class operating-point F1 (with recorded confidence and IoU), while COCO
  AP is reported separately. Per-epoch immutable Lightning checkpoints and
  atomic progress records keep preempted resumes consistent; preempted Optuna
  trials no longer overwrite completed trial directories. Campaign jobs cover
  v8 and v11 detection and segmentation, verify a shared real-domain cohort,
  keep one instance per COCO mask annotation, and sync variants by `run_name`.
  Saved audit findings are stable codes, CLI/report prose is translated, and
  container builds stamp the source commit into provenance.

### Changed
- **Per-image calibration index for crops without a visible dish.** A new
  English-only file `calibration_index.csv` with the headers `image,um_per_px`
  supplies one µm/px per image basename, selected with `--calibrate index
  --calibration-index PATH` (the flag alone selects index mode). Two photos in
  one batch can now keep different scales; a photo absent from the index is
  recorded with `image_not_in_index` and no scale instead of silently using a
  batch-wide fallback. The strict reader rejects the legacy Spanish filenames
  and headers with an error naming the canonical format, refuses duplicate or
  invalid rows, and never searches for or rewrites the file. Legacy index files
  remain unread and untouched; schema v2 is unchanged.
- **English-only run artifacts; legacy readers removed** (ADR-0004; supersedes ADR-0001, ADR-0002 and ADR-0005). All machine tokens in `run.json`, `detections.csv`, logs, Viewer exports and report ids are now English: `sucio` → `dirty`, `placa` → `plate`, `visor_` → `viewer_`, `resumen` → `summary`. New files carry `schema_version: 2`; `load_run` rejects version 1 and pre-canonical folders (`summary.json` + `centroids.csv`) are no longer read. Old runs and archived reports stay on disk but cannot be opened by this version. Scripts must switch to the English tokens; there is no compatibility option. The full replacement list is in `docs/specs/run-artifact-vocabulary.md`.
- Tiling has one owner: `polyx.core.detect.tiling` holds the decision
  (`decide_tiling`), the typed plan and the one sentence each image's plan is
  phrased with. The mode is English at every boundary (`auto`/`always`/`never`);
  a saved legacy spelling still loads, the CLI keeps those as hidden
  deprecated aliases, and `run.json` now records `tiling_plan` per image/model
  while keeping the legacy `tiling` string for old readers.
- **Detection CSV headers are one canonical English schema on every door**
  (ADR-0004, a schema change). The Results page and the Viewer stops writing
  Spanish headers: both export named subsets of `DETECTION_COLUMNS`
  (`PAGE_COLUMNS`, `VIEWER_COLUMNS`), and the Viewer's `size_source` values are
  `mask`/`box` instead of `mascara`/`caja`. Scripts keyed on `modelo`,
  `clase`, `imagen` or `talla_origen` must switch to `model`, `class`, `image`
  and `size_source`; there is no compatibility option. The page's `image`
  cell is the path the run recorded, as in `detections.csv`, not the bare file
  name.
- One operating-point module owns precision/recall/F1
  (`core/detection_run/operating_point.py`). The Results KPIs read the match
  the run decided under the localisation criterion; the confidence advice and
  the report's second F1 column use the with-class criterion. The Compare
  table scores under the localisation criterion and its F1 column names it, so
  the two pages cannot show an unlabelled figure that means something else.
  Every label on a surface names the criterion it shows. The duplicate
  helper in `core/report_metrics.py`, the arithmetic in the Compare page and
  the report's threshold sweep were removed.
- Re-reading the ground truth of a finished run now goes through the
  detection-run engine (`reload_ground_truth` in
  `core/detection_run/reload.py`) instead of a GUI copy of the sequence. The
  reload measures with the run's own shape path, sizes each label with that
  photo's recorded calibration, matches with the run's own rule (mask IoU when
  the run did) and records that rule next to the new counts. It republishes the
  affected images in `run.json`, so the reloaded views and the run record can no
  longer disagree. `gui/detector/recargar_gt.py` is gone; the Results page is a
  progress dialog and one call.
- Model aliases are stable run-folder keys, separate from the label the
  interface shows: the GUI writes `model_1`/`model_2`, the CLI writes the
  weights stem unless `--alias` names one, and `run.json` records
  `alias_generated` so the report translates only a generated label. Changing
  the interface language no longer renames a run folder; old folders still load
  and show their recorded alias verbatim.
- The per-image result carries the match the run decided (`MatchResult` with
  the TP/FP/FN/misclassified indices) and the rule behind it (`iou_thr`, box
  or mask IoU). The Errors page, the Results KPIs, the report's error section
  and both threshold sweeps read that decision instead of re-matching with a
  rule of their own; `run.json` records the rule per image/model and
  `load_run` rebuilds the match from it. Old run folders still load.
- The HTML report package is internal: its public surface is
  `generate_report` plus the section catalog (`SECTIONS`, `IDS_SECTIONS`,
  `PRESETS`, `anchor`), the batch precompute is a named immutable result and
  every section has one uniform builder. Rendered reports are byte-identical.
- `ruff-pre-commit` bumped to `v0.17.2` to match the locked ruff version.

### Security
- `hydra-core` 1.3.2 → 1.3.8 and `lightning` 2.5.2 → 2.6.6, clearing the
  advisories that failed `dependency-review` (GHSA-2cp2-2r3c-7p7r and
  GHSA-c3wx-c55w-pxjq for Hydra; GHSA-qqmf-gpg7-g8gw, PYSEC-2026-3624 and
  PYSEC-2026-3972 for Lightning). `hydra-optuna-sweeper` 1.3.0 still requires
  `hydra-core>=1.1.0.dev7` and `optuna<3.0.0`, so the Optuna 2.10.1 pin is
  unchanged.

## [2.0.0] - 2026-10-26

### Added
- Headless engines and CLIs: `polyx-train`, `polyx-eval`, `polyx-detect`,
  `polyx-report`, plus `polyx-download-models` for the non-bundled weights.
- Python packaging: `pyproject.toml` with hatchling, src layout, console
  scripts, optional extras `[gui]`, `[dev]`, `[sam2]`.
- Self-contained Spanish-remnant scanner (`tests/english_scan.py`) backing
  the four English-convention gates without LibCST.
- CI wheel build + smoke-test job (build, install wheel in a clean venv,
  import + asset check + `--help` on every headless CLI + `[gui]` import).
- Publish workflow using PyPI Trusted Publishing (OIDC), triggered by `v*` tags.
- `LICENSE` (MIT) and `.env.example`.

### Changed
- Package layout: source moved to `src/polyx/`; tooling under `tools/`
  (`tools/audit`, `tools/manual`, `tools/packaging`); documentation under
  `docs/`; end-user double-click scripts under `install/`.
- Bundled assets (`polyx.ico`) now ship inside the wheel.
- Single source of truth for the version (`src/polyx/__init__.py`,
  read by hatchling).
- PyPI metadata: urls, classifiers, keywords, author email, PEP 639 license.
- README install section documents `pip install polyx-microplastics[gui]`.

### Removed
- LibCST translation codemods, harness, manifests and the `libcst` pin.
- 12.4 MB of generated manual HTML from version control (restored after
  review: a ZIP download must carry the manual offline).
- `auditar_traduccion.py` (moved to `tools/audit/audit_translations.py`).

### Fixed
- Per-image metrics in rebuilt reports use the real `ap_class_index`.
- `empaquetar_para_llevar.bat` shipped an offline bundle without program
  code (it robocopy'ed a `polyx\` directory that no longer exists).
- `auditar_traduccion.py` walked a nonexistent `polyx/` path and always
  reported "-4 translated".
- Manual project-tree figure described the pre-rename layout.