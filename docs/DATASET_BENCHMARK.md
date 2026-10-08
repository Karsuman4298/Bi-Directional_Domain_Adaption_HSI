# Separate multi-dataset experiment scripts

This workflow adds new files only. Existing Houston launchers, training scripts,
model files and reporting code are unchanged. Downloads, preprocessing, training
and evaluation happen when you launch the new scripts on your server. No local
experiment results are supplied or fabricated.

## Run on the GPU server

Copy this checkout, including `hsi_benchmark/` and all the new scripts, to the
server. Use a separate Python 3.10 or 3.11 environment for the legacy external
methods. Install **CUDA-compatible PyTorch and torchvision** appropriate for the
server first, then install the other dependencies:

```bash
python -m pip install -r requirements-benchmark.txt
```

The dependency file preserves Cleanlab's old 1.x API used by the existing CLDA
and SCLUDA recipes. It also includes the Hub downloader, HDF5 support, PyG,
TensorBoard, Torchsampler and OpenCV's guided filter. Avoid upgrading this
environment while experiments are running.

Run everything with one command:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all_dataset_experiments.sh
```

Defaults: **12 implemented methods × 8 directions × 2 preprocessing settings ×
3 seeds = 576 runs**, each requesting 120 epochs. Jobs run sequentially on the
selected GPU to avoid GPU-memory contention. `CUDA_VISIBLE_DEVICES=2` selects
physical GPU 2; each method uses logical CUDA device 0 inside that selection.
`PYTHON_BIN=/path/to/environment/bin/python` selects a Python interpreter.

Separate PCA-only and no-PCA scripts are also provided:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_dataset_experiments_with_pca.sh
CUDA_VISIBLE_DEVICES=0 bash run_dataset_experiments_without_pca.sh
```

These use `benchmark_runs/pca30/` and `benchmark_runs/no_pca/`, respectively.
Use the all-experiments script for **one combined report containing both modes**.
All three scripts accept the same options; see the examples below.

## Datasets, directions and classes

The scripts automatically download the selected `.mat` files from
[Tanishq165/HSI_Datasets](https://huggingface.co/datasets/Tanishq165/HSI_Datasets).
Only the eight requested scenes are fetched, rather than the entire collection.
The first run resolves the requested Hub revision to a commit and records it;
resumes download from that same commit. The default download directory is
`benchmark_data/<commit>/`. Hub downloads reuse already downloaded files.

| Pair | Directions | Shared classes used |
|---|---|---:|
| Loukia, Dioni | Loukia → Dioni; Dioni → Loukia | 12 |
| Pingan, Qingyun | Pingan → Qingyun; Qingyun → Pingan | 6 |
| Utopia, NiliFossae | Utopia → NiliFossae; NiliFossae → Utopia | 4 |
| PaviaU, PaviaC | PaviaU → PaviaC; PaviaC → PaviaU | 7 |

PaviaU and PaviaC resolve to the Hub's `Pavia_University/` and `Pavia_Center/`
folders. `NiliFossae` is the Hub spelling of Nilifossae.

This is a **closed-set shared-class experiment**. Scene-specific classes and
background receive the ignored label. The adapter remaps shared classes to the
same contiguous IDs in both scenes; identical raw numbers are not assumed to
represent the same class. For example, Utopia class 2 and NiliFossae class 4 both
map to Bassanite. Utopia and NiliFossae each contain nine named classes, but only
four class names overlap. The seven Pavia classes exclude scene-specific water,
tiles, gravel and painted metal sheets.

Pingan's Road is grouped with Qingyun's Asphalt road, and Pingan's Seawater is
grouped with Qingyun's Water. These explicit semantic assumptions appear in the
report. To use a different definition, supply `--class-map /path/to/maps.json`.
The JSON must have the same four keys as `hsi_benchmark/catalog.py`, with values
containing `datasets`, `classes` and optionally `assumptions`. Each `classes` row
is `["Class name", first_scene_original_ID, second_scene_original_ID]`. Class
names and original IDs must be unique within each pair. An absent selected class
causes a visible preparation failure rather than an invented accuracy.

Both standard MAT files and MATLAB v7.3/HDF5 files are supported. The loader
finds the cube and GT file, validates the numeric arrays and spatial alignment,
and masks nonfinite spectra before class mapping.

## PCA and no-PCA preprocessing

Every method receives the **same prepared arrays** for a direction and mode,
reused across all seeds. The preprocessing seed defaults to 0 and is independent
of the model-training seeds.

1. Align the two spectral dimensions if their native band counts differ.
2. Select up to 100,000 finite source spectra without consulting their labels.
3. Fit a per-band standard scaler on those source spectra.
4. For PCA runs, fit **`PCA(n_components=30)`** on the standardized source sample
   and transform both scenes using that **same source-fitted basis**.
5. For no-PCA runs, keep all aligned spectral bands and apply the same
   source-fitted standardization, without input dimensionality reduction.

Each direction has its own source-fitted transform. Reverse-direction runs
start fresh models; they do not evaluate a forward checkpoint backward.
Target class labels do not fit the scaler or PCA. Fit indices, scaler parameters,
PCA components, explained variance and input file hashes are saved for inspection.

Pavia and the Mars scenes have unequal native band counts. Existing methods
expect a common input width. The default `--band-alignment resample` linearly
interpolates the larger spectrum onto `min(source bands, target bands)` positions
on a normalized **band-index** axis. This is an explicit numerical alignment;
without wavelength metadata it does not establish equal physical wavelengths.
Both modes use this same alignment before their shared preprocessing.

Alternative policies:

```bash
# Keep the first min(source bands, target bands) bands in each scene.
bash run_all_dataset_experiments.sh --band-alignment truncate --output benchmark_runs/truncated

# Fail preparation for unequal band counts instead of altering the spectra.
bash run_all_dataset_experiments.sh --band-alignment error --output benchmark_runs/strict_bands
```

"Without PCA" refers to **input-band reduction**. MLUDA's original guided image
adaptation retains its internal PCA guidance. Its guide uses three components
in this adapter because OpenCV requires a one- or three-channel guide. If the
goal is to ablate every internal PCA operation in MLUDA as well, that is a
different method ablation from the input-band comparison implemented here.

SSWADA's spectral convolution originally requires a multiple of 16 input bands.
The separate adapter uses `gcd(input bands, 16)` groups, allowing 30-band PCA
inputs and arbitrary aligned native counts. Inputs still have exactly 30 bands
in PCA mode. GAHT retains its existing internal channel padding.

The MDGTnet adapter splits the bands into two nonempty spectral partitions at
the midpoint. The existing Houston adapter's `[0, bands]` partition creates an
empty convolution branch. The separate adapter also preserves MDGTnet's batch
dimension when the final evaluation batch contains one pixel.

## Methods and scope

The default suite includes:

`GAHT`, `BiDA`, `AgentBiDA`, `SelfAttentionAgentBiDA`, `TSTnet`, `CLDA`,
`SCLUDA`, `MLUDA`, `SSWADA`, `CACL`, `PCADA`, `MDGTnet`.

The agent methods use the existing `_fix` implementations, as in the restored
comparison launcher. The standalone adapters reuse each method's existing
training loop, loss and optimizer. They change data access, input/class dimensions,
memory allocation and metric export in an isolated worker process. No downloaded
model code or new approximation of a named baseline is substituted.

Some older shell scripts list `cnn3d`, `ablstm`, `dffn`, `m3ddcnn`, `rssan`,
`speformer` and `ssftt`, but their implementations/imports are absent in this
checkout. They cannot be run as implemented baselines and are not advertised as
successful experiments. The new suite covers every implemented method listed above.

Equal input preprocessing does **not** make the entire scientific protocol equal:
the retained recipes use different source sampling, patches, batches and schedules.
PCADA queries target labels; the existing MDGTnet adapter uses target labels for
supervised training. Other methods use domain adaptation or source-only training.
Several recipes select their checkpoint on target accuracy. These distinctions are
disclosed in `results.md`; do not interpret all columns as one unsupervised protocol.
Core methods, TSTnet and SSWADA evaluate each epoch in this workflow, including
short server smoke runs. CLDA may stop early according to its pseudo-label logic;
its result records the actual completed epoch count.

## Results and recovery

The full-suite report is:

```text
benchmark_runs/all_datasets/results.md
```

It contains all **16 direction/mode sections**, with OA, AA, Kappa ×100 and a
per-class recall table for every method. Each metric reports **mean ± sample
standard deviation over all configured seeds**. A single seed is explicitly
labeled `n=1`. Partial or failed sets show `pending`; no zero scores, best-seed
selection or averaging over only successful seeds is used.

```text
benchmark_runs/all_datasets/
├── configuration.json
├── results.md
├── failures.json
├── source_snapshot/        # frozen Python sources used by the workers
├── prepared/<source>_to_<target>/<mode>/
│   ├── manifest.json
│   ├── transform.npz
│   ├── source_image.npy
│   ├── source_labels.npy
│   ├── target_image.npy
│   └── target_labels.npy
└── runs/<source>_to_<target>/<mode>/<method>/<seed>/
    ├── training.log
    ├── status.json
    ├── result.json         # only when actual evaluated scores were exported
    ├── last_epoch.pt       # methods with existing recovery support
    └── checkpoints/        # available best/final weight files
```

Results include a validated confusion matrix, class support, input-band count,
class names and checkpoint-selection metadata. The report shows actual evaluated
pixel counts: MLUDA's existing evaluator drops its incomplete final batch, and
the core recipes retain reflected border labels. Their evaluation populations
can therefore differ from the eligible original target counts.

The report refreshes after each run, including failures. Failures are recorded
and the suite proceeds to the next job; the final exit code is nonzero if any
configured job failed. Logs link from the Markdown report.

Resume with **the same options and output directory**:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all_dataset_experiments.sh --resume
```

Completed, validated runs are skipped. Existing epoch recovery is reused for
GAHT, BiDA, both agent methods, TSTnet, CLDA, SCLUDA, MLUDA, SSWADA and CACL.
PCADA and MDGTnet restart unfinished runs from the beginning because their
existing adapters do not save full epoch recovery state. Code, settings, dataset
hashes and package versions are checked; use a new output directory for a changed
experiment. Resume is not a promise of bitwise-identical GPU results.

Worker source files are frozen in `source_snapshot/` at launch, so later edits to
the checkout cannot change the methods halfway through an active suite. Leave
the saved snapshot unchanged; its contents are checked when resuming.

Rebuild the report without downloading or training:

```bash
python run_dataset_benchmark.py --report-only --output benchmark_runs/all_datasets
```

## Useful server commands

Inspect the job plan without downloading data, importing PyTorch or training:

```bash
bash run_all_dataset_experiments.sh --dry-run
```

Run the supplied offline unit checks on the server. They use small synthetic MAT
fixtures to check class mapping, shared PCA, HDF5 orientation, cache validation,
all twelve entry-point adapter locations, and incomplete-seed reporting:

```bash
python -m unittest discover -s tests -p 'test_dataset_benchmark.py'
```

Download and preprocess all requested data without training:

```bash
bash run_all_dataset_experiments.sh --prepare-only
# Then run the same experiment, using the prepared output:
CUDA_VISIBLE_DEVICES=0 bash run_all_dataset_experiments.sh --resume
```

Run a small **server** integration check before the full suite. A one-epoch run
tests execution and export; it is not a convergence or full-algorithm check:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all_dataset_experiments.sh \
  --pairs Utopia-NiliFossae --direction forward --seeds 2100 --epochs 1 \
  --output benchmark_runs/server_smoke
```

Customize epochs/seeds or use a subset of the implemented methods:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all_dataset_experiments.sh \
  --epochs 200 --seeds 2100 2101 2102 \
  --models GAHT BiDA TSTnet CLDA SCLUDA MLUDA SSWADA CACL PCADA MDGTnet \
  --data-dir /scratch/hsi_data --output /scratch/hsi_results
```

Source normalization/PCA runs in chunks. Large target patch arrays are disk-backed
and shared with PyTorch instead of copied in full into RAM. Temporary patch arrays
are deleted after each worker, and legacy scratch workspaces are deleted by default.
Available final/best weights are copied to the run folder before cleanup. Use
`--keep-workspaces` to retain additional legacy files. Raw-band Pingan/Qingyun runs
can still need tens of GB of temporary disk space; provide ample server RAM and
scratch storage (at least about 80 GB free scratch is a reasonable starting point).

The new experiment scripts and tests were not run locally, and no datasets were
downloaded locally. The server commands above are the validation steps to run there.
