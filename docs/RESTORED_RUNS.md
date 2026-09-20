# Restored recipes: tables and recovery

These changes preserve the restored architectures, losses, optimization schedules,
and evaluation populations. They add reporting and recovery, fix score-export errors,
and default to **four agents**, 120 epochs and seeds 2100/2101/2102. They do not
establish reproduction of the published paper or a common evaluation protocol.

Use the environment that ran restored BiDA successfully. Dependencies for external
methods still need to be present; failures remain visible and do not count as scores.

```bash
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=0 python -u run_restored_all.py \
  --direction both --epochs 120 --seeds 2100 2101 2102 --num-agents 4 \
  --output restored_runs/agents4_120
```

To continue that same experiment after stopping, use exactly the same command with
`--resume` appended. Completed runs with valid score files are skipped. Interrupted
runs restore `last_epoch.pt` and continue at the next epoch. Work in the unfinished
epoch is repeated. This is epoch-boundary recovery, not minibatch recovery.

All ten adapters have recovery hooks. These save model weights/buffers, persistent
optimizers, EMA/scheduler state where present, random states, loader generators,
best-checkpoint information, and required cached pseudo-label/prototype data.
TSTnet, SCLUDA and MLUDA recreate their optimizer each epoch in the restored recipe;
that behavior is preserved rather than silently changed. GPU nondeterminism and
legacy preprocessing mean bitwise-identical full-model reruns are not guaranteed.
CPU tests verify checkpoint/RNG round trips and an interrupted core training run.
Full resumed external GPU training has not been verified.

Model implementations: GAHT and BiDA use `main.py`; the proposed models use the
historical `_fix` entry points. Default width 64, depth 3 and effective heads 8
are unchanged. Four agents are explicit in the command.

## Existing runs

An older timestamped output can be resumed with `--output` pointing at it, provided
its direction, model list, seed list, epochs and agent count match exactly. Valid
completed legacy scores are retained and explicitly labeled as legacy imports with
unverified provenance. Invalid scores (including OA above 100) are rerun.
The launcher preserves the original configuration as `legacy_configuration.json`.

**Old interrupted runs have no epoch checkpoints and restart from epoch one.**
Two-agent results cannot be imported into a four-agent run. There is no conversion
of an old best-model checkpoint into a complete optimizer/RNG resume checkpoint.

For new runs, changes to code, Houston data, recorded package versions or settings
are rejected on resume. Use a new output directory for a different experiment.
Do not update the code while the suite is running. The launcher locks the checkout
because legacy exporters share repository-relative paths.

## Outputs

Each `<direction>/<model>/<seed>/` directory contains an appended `training.log`,
`status.json`, validated `result.json` when available, and `last_epoch.pt` after the
first completed epoch. Core best checkpoints and CACL's best file also live in this
run directory. Other external historical checkpoints may additionally remain in
those methods' original output folders. Existing legacy result files are archived,
not counted as a newly successful run.

Tables refresh after every run, including failures:

- `tables/13to18_comparison.md`, `.csv`, `.tex`
- `tables/18to13_comparison.md`, `.csv`, `.tex`

Rows: seven class recalls, OA, AA, Kappa x100; columns: model names. Each metric
requires exactly all configured seeds and reports mean ± sample SD (ddof=1).
Missing/failed runs show `pending`, with a completed-seed count. No fabricated zeros,
best-seed selection, or averaging over only successful seeds. A one-seed experiment
is labeled `n=1` rather than inventing a standard deviation.

Regenerate tables without training:

```bash
python restored_reporting.py --root restored_runs/agents4_120
```

Markdown tables also disclose sample counts and checkpoint-selection rules.
Different counts or selection rules mean columns are **descriptive restored-recipe
results**, not a validated fair comparison. Equal class counts alone would still
not prove identical evaluation coordinates. Legacy imports may lack this metadata.
LaTeX uses `booktabs` and `graphicx`.

The demonstration in `restored_table_demo.md` uses synthetic fixture counts only;
it is not a prediction of model accuracy.

## Reporting corrections

- SCLUDA/MLUDA use their actual final `labels` and `predict` arrays, replacing
  undefined exporter variables; OA denominator is the evaluated prediction count.
- CLDA unpacks all four outputs returned by its evaluation function.
- TSTnet/SSWADA no longer multiply a percentage OA by another factor of 100.
- Exported class IDs use seven explicit classes; metrics are recomputed from the
  confusion matrix and validated before inclusion in a table.
- TSTnet exports its best evaluated result even when OA does not exceed 50%; this
  removes a reporting threshold, without modifying its training objective.

The restored CLDA recipe can exit training early if its pseudo-label pool lacks
classes. Recovery does not override that algorithmic behavior. Its reported rule
is `after_training_loop`, not a claim that every requested epoch was executed.
