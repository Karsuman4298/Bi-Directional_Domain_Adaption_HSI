"""One Markdown report containing every configured direction, mode and method."""
import json
from pathlib import Path

from .catalog import PROTOCOLS, UNAVAILABLE_METHODS
from .metrics import validate


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def cell(values, required):
    if len(values) != required:
        return "pending"
    return f"{values[0]:.2f}"


def write_report(root, configuration):
    root = Path(root)
    seeds = configuration["seeds"]
    if len(seeds) != 1:
        raise ValueError("This benchmark requires exactly one training seed; use a new output directory for a single-seed run")
    methods = configuration["models"]
    lines = [
        "# Multi-dataset HSI experiments", "",
        "Data source: [Tanishq165/HSI_Datasets](https://huggingface.co/datasets/Tanishq165/HSI_Datasets).", "",
        f"Pinned Hub commit: `{configuration.get('hf_revision', 'not resolved')}`. "
        f"Requested epochs: {configuration['epochs']}; training seed: {seeds[0]}.", "",
        "PCA runs use exactly `PCA(n_components=30)`. One source-fitted transform is shared "
        "by all methods for a direction. No-PCA runs retain the aligned spectral bands.", "",
        f"Band alignment policy: `{configuration['preprocessing']['band_alignment']}`. "
        "For unequal spectra, resample means linear interpolation on normalized band index; "
        "truncate means keeping the first min(source bands, target bands) bands. "
        "Neither establishes physical wavelength correspondence.", "",
        "Metrics are percentages; Kappa is multiplied by 100. Scores come from one training "
        "seed with no averaging or standard deviation. Failed or missing runs "
        "show pending and contribute no invented scores.", "",
        "These are adapted existing recipes with a common input preprocessing and closed-set "
        "class mapping. Their sampling, patch sizes, optimization, target supervision and "
        "checkpoint selection differ. PCADA uses queried target labels; the existing MDGTnet "
        "adapter uses true target labels. MLUDA retains internal PCA for its guided image "
        "adaptation even in the no-input-PCA condition (three-channel OpenCV guidance).", "",
        "SSWADA's grouped spectral convolution uses gcd(input bands, 16) groups to support "
        "30-band inputs and unequal native band counts. The legacy 16-group setting is retained "
        "when the input band count is divisible by 16.", "",
        "MDGTnet uses two nonempty spectral partitions split at half the input bands; "
        "the existing adapter's empty first partition is incompatible with its convolution. "
        "Its output preserves the batch dimension for a single final evaluation pixel.", "",
    ]
    launcher_error = read_json(root / "launcher_error.json")
    if launcher_error:
        lines += ["Launcher error: " + str(launcher_error.get("error", "Unknown error")), ""]
    preparation_errors = read_json(root / "preparation_errors.json") or []
    for error in preparation_errors:
        lines += ["Preparation error: " + str(error), ""]
    failures = []
    for task in configuration["tasks"]:
        for mode in configuration["modes"]:
            lines += [f"## {task['source']} → {task['target']} — {'PCA (30 bands)' if mode == 'pca30' else 'Without input PCA'}", ""]
            prepared = read_json(root / "prepared" / task["direction"] / mode / "manifest.json")
            if prepared:
                lines += [
                    f"Native source/target bands: {prepared['native_bands'][0]}/{prepared['native_bands'][1]}; "
                    f"input bands for every method: {prepared['n_bands']}; shared classes: {prepared['n_classes']}.", "",
                    "Eligible target counts: " + ", ".join(
                        f"{name}={count}" for name, count in zip(task["class_names"], prepared["target"]["support"])
                    ) + ". Actual evaluated counts appear below; some legacy recipes omit a final incomplete batch "
                    "or include reflected border labels.", "",
                ]
            for assumption in task["assumptions"]:
                lines += ["Class mapping assumption: " + assumption, ""]
            lines += [
                "| Method | Complete seeds | OA | AA | Kappa ×100 | Evaluated pixels | State |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
            records = {}
            for method in methods:
                scores = []
                failed = []
                states = []
                for seed in seeds:
                    path = root / "runs" / task["direction"] / mode / method / str(seed)
                    status = read_json(path / "status.json") or {}
                    result = read_json(path / "result.json")
                    state = status.get("state", "pending")
                    if state == "complete" and result:
                        try:
                            validate(result, len(task["class_names"]))
                            expected = configuration["run_fingerprint"] + f"/{task['direction']}/{mode}/{method}/{seed}"
                            if result.get("run_fingerprint") != expected or status.get("run_fingerprint") != expected:
                                raise ValueError("Result provenance mismatch")
                            scores.append(result)
                        except (KeyError, ValueError, TypeError):
                            state = "invalid result"
                    if state in ("failed", "interrupted", "invalid result"):
                        failed.append(seed)
                        failures.append((path, state, status.get("error") or "Inspect training.log"))
                    states.append(state)
                records[method] = scores
                counts = sorted({sum(score["support"]) for score in scores})
                count_text = ", ".join(map(str, counts)) if counts else "pending"
                state_text = "complete" if len(scores) == len(seeds) else (
                    "failed seeds: " + ", ".join(map(str, failed)) if failed else ", ".join(sorted(set(states)))
                )
                metrics = [cell([score[key] for score in scores], len(seeds)) for key in ("OA", "AA", "Kappa")]
                lines.append(f"| {method} | {len(scores)}/{len(seeds)} | {' | '.join(metrics)} | {count_text} | {state_text} |")
            lines += ["", "Per-class recall (%):", "",
                      "| Class | " + " | ".join(methods) + " |",
                      "|---|" + "---:|" * len(methods)]
            for class_id, name in enumerate(task["class_names"], 1):
                values = [cell([score["classes"][str(class_id)] for score in records[method]], len(seeds)) for method in methods]
                lines.append("| " + name + " | " + " | ".join(values) + " |")
            lines += ["", "Original GT IDs → common class IDs:", "",
                      f"| Common ID | Class | {task['source']} original ID | {task['target']} original ID |",
                      "|---:|---|---:|---:|"]
            for index, (name, source_id, target_id) in enumerate(zip(task["class_names"], task["source_ids"], task["target_ids"]), 1):
                lines.append(f"| {index} | {name} | {source_id} | {target_id} |")
            lines.append("")
    lines += ["## Recipe notes", "", "| Method | Existing training/evaluation recipe |", "|---|---|"]
    for method in methods:
        lines.append(f"| {method} | {PROTOCOLS[method]} |")
    lines += ["", "Baseline names referenced by old launchers with implementations absent in this checkout: "
              + ", ".join(UNAVAILABLE_METHODS) + ". No results are claimed for these names."]
    lines += ["", "## Failed or interrupted runs", ""]
    if not failures:
        lines.append("None recorded. Pending rows have not produced a complete set of validated scores.")
    for path, state, error in failures:
        relative = path.relative_to(root).as_posix()
        error = str(error).replace("\n", " ").replace("|", "/")
        lines.append(f"- [{relative}]({relative}/training.log): {state}; {error}")
    lines += ["", "Per-run JSON includes the confusion matrix, class support and actual checkpoint selection. "
              "Configuration, preprocessing manifests, logs and available checkpoints are stored beside this report.", ""]
    temporary = root / "results.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "results.md")
