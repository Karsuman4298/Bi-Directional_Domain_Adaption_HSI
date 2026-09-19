# Controlled Houston domain-adaptation experiments

The supported entry point is `python -m experiments.run`, wrapped by
`run_fair_houston.sh`. These are **corrected, standardized repository recipes**,
not an assertion of exact reproduction of each publication. Do not combine old
checkpoint-selected scores or manually entered JSONs with these results.

## Environment and server commands

Copy the updated source tree to the server before running these commands. Changes
in this workspace have not been committed or pushed automatically. Use a fresh
Python 3.11 virtual environment; do not repair the old environment by repeatedly
upgrading/downgrading NumPy and compiled extensions.

```bash
cd /path/to/IEEE_TCSVT_BiDA
nvidia-smi
bash setup_fair_environment.sh
source .venv-fair/bin/activate
python -m pytest -q tests/test_fair_protocol.py tests/test_optional_refinement.py
```

The setup script pins PyTorch 2.8.0 and defaults to its CUDA 12.6 wheel. Choose the
wheel compatible with the server driver/GPU **before installation**. For a
Blackwell GPU / CUDA 12.8-capable driver use:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_fair_environment.sh
```

For a CPU-only environment, install PyTorch's CPU wheel manually in a fresh venv,
install `requirements-fair.txt`, then use `python -m experiments.preflight --cpu`.
PyTorch publishes the supported wheel commands in its
[versioned installation instructions](https://docs.pytorch.org/get-started/previous-versions/).
Cleanlab uses its modern 2.7 API; the old `cleanlab==1.0.1` compatibility patches
are not used. The benchmark does not need imagecodecs, hdf5storage, torchvision,
torch-geometric, or a globally modified PYTHONPATH.

Run both-direction smoke tests on the chosen GPU (two batches/model; no scores):

```bash
CUDA_VISIBLE_DEVICES=3 bash run_fair_houston.sh both smoke
```

Run the main comparisons, including all available UDA baselines:

```bash
CUDA_VISIBLE_DEVICES=3 bash run_fair_houston.sh both full
```

Or use two terminals / two GPUs:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=2 bash run_fair_houston.sh 13to18 full
# Terminal 2
CUDA_VISIBLE_DEVICES=3 bash run_fair_houston.sh 18to13 full
```

Do not run `both full` concurrently with the same direction on another GPU.
Completed matching runs are skipped; interrupted runs restart from epoch one.
Logs remain under `fair_results/logs`. `set -euo pipefail` preserves Python errors
through tee; a missing baseline stops the command instead of creating a partial
comparison table.

Defaults: 120 epochs and seeds 2100, 2101, 2102 for both full benchmarks
and component ablations. To change the budget,
use a new result directory and apply the same budget to every method:

```bash
OUTPUT=fair_results_100 EPOCHS=100 SEEDS='2100 2101 2102' \
CUDA_VISIBLE_DEVICES=3 bash run_fair_houston.sh both full
```

MMD starts in epoch one, including a 100-epoch run. Three seeds are supported;
five are the default. More independent seeds provide more informative uncertainty
estimates. Do not rerun/select seeds according to target test performance.

## Component and agent-count ablations

```bash
CUDA_VISIBLE_DEVICES=3 bash run_fair_houston.sh both ablation
```

This evaluates all three BiDA architectures with the same settings under:
full objective, source only, no MMD, no distillation, no consistency, and no
class-wise pairing. Full runs already present are reused. Component ablations
produce one combined Markdown/CSV/LaTeX table per direction. They are not replaced
by comparing three full architectures alone.

For an independently reported agent-count study:

```bash
for agents in 1 2 4; do
  NUM_AGENTS="$agents" OUTPUT="fair_results_agents${agents}" \
  MODELS='BiDA AgentBiDA SelfAttentionAgentBiDA' CUDA_VISIBLE_DEVICES=3 \
  bash run_fair_houston.sh both full
done
```

Keep every agent-count result in the report. Do not select the count using the
held-out target scores and then call those same scores an independent test.

## Tables and traceability

`fair_results/tables/` contains class recall for the seven classes, OA, AA and
Kappa x 100, expressed as mean ± **sample** standard deviation (`ddof=1`).
The schema records class IDs 1–7 in the paper's order. Metrics are recomputed from
saved predictions and original coordinates. The aggregator rejects missing seeds,
ambiguous duplicate runs, mixed code/data/budgets, and mismatched BiDA-family
settings. It never takes the best epoch or best seed.

Regenerate the complete tables without training:

```bash
python -m experiments.table --source Houston13 --target Houston18
python -m experiments.table --source Houston18 --target Houston13
```

To aggregate a deliberate subset, specify exactly the models/seeds trained:

```bash
python -m experiments.table --source Houston13 --target Houston18 \
  --models GAHT BiDA AgentBiDA SelfAttentionAgentBiDA --seeds 2100 2101 2102
```

`*.paired.json` contains each proposal's per-seed differences from BiDA and
95% paired t intervals for OA/AA/Kappa. These are descriptive intervals across
training seeds, not spatially independent pixel tests, and are not corrected for
multiple comparisons. Report all three metrics and the rare-class recalls;
positive OA alone does not demonstrate a universal improvement.

Each run saves its final checkpoint, predictions, original train/validation/test
coordinates, dataset hashes, source-code hash, package versions, hyperparameters,
and per-epoch losses. Changing code creates a different run ID. Keep a frozen code
snapshot for a complete study. Smoke runs cannot enter publication tables.

## Common experimental protocol

- Data: the supplied 48-band scenes. Verify source/target foreground class counts
  against paper Table IV: Houston13 = 2,530, Houston18 = 53,200. Image dimensions
  in these files are 210 × 954. Evaluate each original labeled coordinate once.
- Input normalization: per-pixel spectral L2 normalization. Only images are padded;
  labels and coordinate populations are never padded. MLUDA additionally performs
  its image-level transformation as described below.
- Source labels: a stratified 95% training / 5% validation split for every method,
  with the same split seed. Validation labels are retained for provenance but not
  used for this fixed-final-epoch selection rule. The random split has spatially
  overlapping patches and should not be presented as spatial holdout validation.
- Target training: the original foreground mask is used, as in this benchmark;
  target **class IDs are not stored in the training dataset or passed to trainers**.
  This is a transductive foreground-mask protocol, not a claim that foreground
  locations are unknown. Target class IDs are used only after training for metrics.
- Checkpoint selection: the predeclared final epoch. No target evaluation, early
  stopping, target-label pseudo-label correction, or target-based LR selection.
- Each epoch cycles the smaller loader for `max(source_batches,target_batches)`
  full minibatches. Approximate image exposure is comparable; numbers of optimizer
  substeps differ by algorithm. Method-specific patches, batch sizes, LR and
  additional transforms are recorded, not silently made identical.
- All three BiDA architectures share patch 13, batch 128, SGD LR .01, 8 heads,
  width 64, 3 layers, 4 semantic tokens, lambda1=.1, lambda2=1, EMA=.999.
  Proposed models default to 4 agents. Baseline/proposal fusion residuals are both
  propagated; unused baseline token parameters are removed.
- BiDA MMD is a finite multi-kernel biased RBF estimator on normalized CLS
  **features**, not 7-class logits. It supports unequal batch sizes. Both domain
  terms are summed. Feature choice is explicit: final CLS features rather than
  all flattened token representations.
- Confident target predictions from the EMA teacher satisfy entropy <= .5 log(7).
  Matched source/target classes in the current minibatch form coupled pairs;
  unavailable classes are skipped, not paired to a wrong class. Unpaired samples
  still contribute to supervised/marginal/consistency losses. Pair sampling is
  within the current batch, not a full-dataset pseudo-label pool.
- Coupled soft-target cross-entropy is jointly differentiated, as in the supplied
  implementation; the EMA teacher is frozen. No second softmax on probabilities.
  Teacher/student get independent center-preserving rotations/flips/Gaussian
  noise (sigma .05), and EMA parameters plus BN buffers are maintained. Random
  clipping is deliberately not used because it can move the center-pixel label.
- Adaptive agent pooling preserves CLS and uses explicit average bins with the
  same semantics on CPU/CUDA/MPS. Historical `_fix` imports alias canonical models.

## External recipe changes and limits

External architectures come from `external_methods`. Their broken standalone
experiment scripts are historical references, not the supported training path.
The new `experiments.external` adapters explicitly implement their objectives;
these standardized recipes require convergence checks on the GPU server and
should be described as **our controlled reimplementations**, not official scores.

| Model | Patch / batch / initial LR | Recipe / corrections |
|---|---|---|
| GAHT | 13 / 128 / .001 | Source-supervised baseline. |
| TSTnet | 12 / 100 / .01 | CNN + GraphSAGE, linear MMD, Wasserstein/GW alignment and CNN/GCN supervision. Mean GraphSAGE is implemented directly with the same neighbor-plus-root operation; all grid boundaries receive graph edges. Persistent SGD momentum; stable device-local OT tensors; final eval returns CNN predictions. |
| CLDA | 5 / 36 / .01 | Two-classifier discrepancy min/max, source supervision, target entropy and confident learning. From epoch 40, every 20 epochs, a source-trained probabilistic SVM and Cleanlab 2.7 prune network pseudo-labels. Uses network argmax pseudo-labels rather than the legacy cluster-refined `obtain_label`; clean CE is applied to retained samples in current minibatches. |
| SCLUDA | 7 / 32 / .01 | Supplied DSAN backbone, local MMD, source/pseudo-target supervised contrastive losses and occupancy loss. Same safe SVM/cleanlab refresh; retained target CE has weight .03. Independent rotation/flip/noise views replace NumPy augmentation. Clean samples are masked within minibatches instead of a separately resampled clean loader. A collapsed one-class pseudo-label pool disables target clean CE until the next refresh. |
| MLUDA | 7 / 32 / .01 | Supplied cross-attention network with LMMD, dual-view contrastive and occupancy losses; PCA/gamma/histogram-guided image preprocessing. Uses **3 PCA guide channels**, because OpenCV rejects the legacy 2-channel guide. Histogram matching is channel-aware and all 48 image bands are filtered. |
| SSWADA | 7 / 128 / .01 | Supplied weighted extractor, three classifiers and two domain discriminators. Three optimization stages; the first discriminator is actually optimized. GRL already reverses feature gradients, so the second reversal/sign error is removed. Source classifier is optimized during supervised stage. |
| CACL | 5 / 36 / .0005 | Supplied encoder/classifier/mapper, domain adversary, source class spatial/spectral prototypes refreshed every 20 epochs, consistent/inconsistent target contrastive and focal terms. Classifier/discriminator return logits for CE; discriminator is trained on detached features before frozen-discriminator feature adaptation. |

External SGD recipes use a persistent optimizer with polynomial LR decay
`lr/(1+10*(epoch-1)/epochs)^.75`; CACL uses constant Adam LR. This removes optimizer
momentum resets. BiDA family uses its shared constant SGD recipe. A fixed common
protocol does not establish that every baseline has received optimal tuning;
report these choices and, for stronger claims, tune with a predeclared source-only
validation procedure or additional development tasks, never target test labels.

Excluded: PCADA (active target annotation), the supplied MDGTnet wrapper (direct
supervised target labels), MSDA (implementation missing), and undefined internal
constructors (cnn3d/ablstm/dffn/m3ddcnn/rssan/speformer/ssftt). Do not rename a
substitute model to fill these columns. Published numbers can be cited separately
for Houston13 → Houston18; no reverse-direction numbers are invented.

CUDA determinism is requested with warnings for unsupported deterministic kernels; exact bitwise equality across devices is not promised. Environment package versions are included in run IDs and must match during aggregation.

The CLDA/SCLUDA refinement SVM uses a reproducible stratified cap of 180 source-training examples per class (all examples for rarer classes), avoiding quadratic SVM fitting on ~50,000 reverse-direction source pixels. This cap affects only the auxiliary SVM; CNN supervision retains the common 95% source split.
