"""Download and run the separate four-pair HSI benchmark on a CUDA server.

No legacy source files are changed. All methods execute sequentially in isolated
working directories, using prepared arrays and process-local compatibility adapters.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from hsi_benchmark.catalog import METHODS, PAIRS, directions
from hsi_benchmark.data import atomic_json
from hsi_benchmark.reporting import read_json, write_report

ROOT = Path(__file__).resolve().parent
ENTRIES = {
    "GAHT": "main.py", "BiDA": "main.py",
    "AgentBiDA": "train_agent_bida_fix.py",
    "SelfAttentionAgentBiDA": "train_self_attn_agent_bida_fix.py",
    "TSTnet": "external_methods/TSTnet/train_tstnet.py",
    "CLDA": "external_methods/CLDA/CLDA_HOUSTON13_2_18.py",
    "SCLUDA": "external_methods/SCLUDA/SCLUDA_Houston.py",
    "MLUDA": "external_methods/MLUDA/MLUDA_hu.py",
    "SSWADA": "external_methods/SSWADA/main.py",
    "CACL": "external_methods/CACL/demo_singleDA.py",
    "PCADA": "external_methods/PCADA/train_pcada_houston.py",
    "MDGTnet": "external_methods/MDGTnet/train_mdgtnet_houston.py",
}


def code_paths():
    paths = [ROOT / name for name in (
        "main.py", "train_pipeline.py", "restored_checkpoint.py", "restored_reporting.py",
        "train_agent_bida_fix.py", "train_self_attn_agent_bida_fix.py",
        "train_dataset_benchmark.py", "run_dataset_benchmark.py",
    )]
    for folder in ("models", "utils", "loss", "external_methods", "hsi_benchmark"):
        paths.extend((ROOT / folder).rglob("*.py"))
    return sorted(set(paths))


def code_hash():
    digest = hashlib.sha256()
    for path in code_paths():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def packages():
    result = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "h5py",
                 "hdf5storage", "huggingface-hub", "cleanlab", "torch-geometric", "tensorboardX",
                 "opencv-contrib-python-headless", "torchsampler", "scikit-image", "einops"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def method_arguments(method, source, target, seed, epochs, agents, workers):
    shared = ["--seed", str(seed)]
    if method in METHODS[:4]:
        arguments = shared + ["--model", method, "--source_name", source, "--target_name", target,
                              "--epoch", str(epochs), "--num_workers", str(workers),
                              "--device", "0", "--log_interval", "1"]
        if method in ("AgentBiDA", "SelfAttentionAgentBiDA"):
            arguments += ["--num_agents", str(agents), "--num_heads", "8"]
        return arguments
    if method == "TSTnet":
        return shared + ["--num_epoch", str(epochs), "--source_name", source, "--target_name", target,
                         "--log_interval", "1"]
    if method == "SSWADA":
        return shared + ["--num_epoch", str(epochs), "--log_interval", "1"]
    if method == "CACL":
        # Existing parser choices are retained. The separate adapter then sets the
        # real scene names, channel count and class count before data/model setup.
        return shared + ["--epochs", str(epochs), "--dataset", "M_Houston", "--in_channel", "48",
                         "--source_name", "Houston13", "--target_name", "Houston18", "--gpu_id", "0"]
    return shared + ["--epochs", str(epochs)]


def snapshot_code(output, paths, expected_hash=None):
    snapshot = output / "source_snapshot"
    digest = hashlib.sha256()
    for source in paths:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        digest.update(source.relative_to(ROOT).as_posix().encode())
        digest.update(content)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError("Source snapshot differs from this configuration: " + str(target))
        else:
            target.write_bytes(content)
    if expected_hash is not None and digest.hexdigest() != expected_hash:
        raise ValueError("Source changed while creating the experiment snapshot; use a new output directory")
    return snapshot


def workspace(destination, paths, snapshot):
    workspace = destination / "workspace"
    for source in paths:
        target = workspace / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(snapshot / source.relative_to(ROOT))
    return workspace


def complete(destination, fingerprint, class_count):
    from hsi_benchmark.metrics import validate
    status = read_json(destination / "status.json") or {}
    result = read_json(destination / "result.json")
    if status.get("state") != "complete" or status.get("run_fingerprint") != fingerprint or not result:
        return False
    try:
        validate(result, class_count)
        return result.get("run_fingerprint") == fingerprint
    except (KeyError, TypeError, ValueError):
        return False


def run_one(destination, task, mode, method, seed, manifest, configuration, args, paths, snapshot):
    fingerprint = configuration["run_fingerprint"] + f"/{task['direction']}/{mode}/{method}/{seed}"
    if args.resume and complete(destination, fingerprint, len(task["class_names"])):
        print("SKIP completed:", destination, flush=True)
        return True
    destination.mkdir(parents=True, exist_ok=True)
    # A partial/stale score must never become a new success if the child fails.
    result_path = destination / "result.json"
    if result_path.exists():
        result_path.rename(destination / f"previous_result_{time.time_ns()}.json")
    work = workspace(destination, paths, snapshot)
    entry = work / ENTRIES[method]
    cwd = work if method in METHODS[:4] else entry.parent
    command = [sys.executable, "-u", str(snapshot / "train_dataset_benchmark.py"), method, str(entry)]
    command += method_arguments(method, task["source"], task["target"], seed, args.epochs,
                                args.num_agents, args.num_workers)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(cwd), str(work), str(snapshot)))
    environment["PYTHONHASHSEED"] = str(seed)
    environment["MPLBACKEND"] = "Agg"
    environment["HSI_BENCHMARK_MANIFEST"] = str(manifest)
    environment["BIDA_RUN_DIR"] = str(destination)
    environment["BIDA_RUN_FINGERPRINT"] = fingerprint
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    status = {"state": "running", "run_fingerprint": fingerprint,
              "command": command, "cwd": str(cwd), "exit_code": None,
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(destination / "status.json", status)
    process = None
    code = -1
    error = None
    try:
        with (destination / "training.log").open("a") as log:
            log.write("\n=== " + status["started"] + " ===\n")
            log.flush()
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            error = "Worker exit code " + str(code)
        else:
            from hsi_benchmark.metrics import validate
            result = read_json(result_path)
            if not result or result.get("run_fingerprint") != fingerprint:
                raise ValueError("Worker did not produce a result for this run")
            validate(result, len(task["class_names"]))
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        status.update(state="interrupted", exit_code=-15, error="Interrupted; rerun with --resume")
        atomic_json(destination / "status.json", status)
        raise
    except (OSError, ValueError, KeyError, TypeError) as exception:
        error = str(exception)
    finally:
        # Recreated deterministically by the patch adapter on resume. Keeping
        # hundreds of raw-band patch buffers would need enormous disk space.
        shutil.rmtree(destination / "patch_cache", ignore_errors=True)
        if not args.keep_workspaces:
            shutil.rmtree(work, ignore_errors=True)
    status.update(state="complete" if code == 0 and error is None else "failed",
                  exit_code=code, error=error,
                  finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(destination / "status.json", status)
    return status["state"] == "complete"


def load_catalog(path, parser):
    catalog = json.loads(Path(path).read_text()) if path else PAIRS
    if set(catalog) != set(PAIRS):
        parser.error("Class-map JSON must contain all four pair keys; see hsi_benchmark/catalog.py")
    for key, value in catalog.items():
        if value.get("datasets") != PAIRS[key]["datasets"] or len(value.get("classes", [])) < 2:
            parser.error("Invalid datasets or too few common classes for " + key)
        seen_names, seen_source, seen_target = set(), set(), set()
        for row in value["classes"]:
            if (len(row) != 3 or not isinstance(row[0], str) or not row[0]
                    or any(type(identifier) is not int or identifier < 1 for identifier in row[1:])
                    or row[0] in seen_names or row[1] in seen_source or row[2] in seen_target):
                parser.error("Require unique names and positive unique original class IDs for " + key)
            seen_names.add(row[0])
            seen_source.add(row[1])
            seen_target.add(row[2])
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "pca", "no-pca"), default="both")
    parser.add_argument("--direction", choices=("both", "forward", "reverse"), default="both")
    parser.add_argument("--pairs", choices=list(PAIRS), nargs="+", default=list(PAIRS))
    parser.add_argument("--models", choices=METHODS, nargs="+", default=METHODS)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2100, 2101, 2102])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--band-alignment", choices=("resample", "truncate", "error"), default="resample")
    parser.add_argument("--fit-pixels", type=int, default=100000)
    parser.add_argument("--chunk-pixels", type=int, default=10000)
    parser.add_argument("--preprocess-seed", type=int, default=0)
    parser.add_argument("--class-map", help="Optional JSON override of the four semantic class maps")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--data-dir", default="benchmark_data")
    parser.add_argument("--output", default="benchmark_runs/all_datasets")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; do not import torch, download or train")
    parser.add_argument("--prepare-only", action="store_true", help="Download and preprocess without training")
    parser.add_argument("--report-only", action="store_true", help="Regenerate results.md from saved scores")
    parser.add_argument("--keep-workspaces", action="store_true", help="Retain legacy scratch files after each worker")
    args = parser.parse_args()
    if args.report_only:
        config = read_json(Path(args.output) / "configuration.json")
        if not config:
            parser.error("No saved configuration found in --output")
        write_report(args.output, config)
        print(Path(args.output) / "results.md")
        return
    if (args.epochs < 1 or args.num_agents < 1 or args.num_workers < 0 or args.fit_pixels < 30
            or args.chunk_pixels < 1 or args.preprocess_seed < 0 or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error("Require positive epochs/agents/chunks, fit-pixels >=30, nonnegative workers, and valid seeds")
    for name in ("seeds", "models", "pairs"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error("Duplicate " + name)
    catalog = load_catalog(args.class_map, parser)
    tasks = list(directions(args.pairs, args.direction, catalog))
    modes = ["pca30", "no_pca"] if args.mode == "both" else ["pca30" if args.mode == "pca" else "no_pca"]
    count = len(tasks) * len(modes) * len(args.models) * len(args.seeds)
    print(f"Planned runs: {count}; output: {Path(args.output).resolve()}/results.md", flush=True)
    if args.dry_run:
        for task in tasks:
            for mode in modes:
                for method in args.models:
                    for seed in args.seeds:
                        print(f"{task['direction']} {mode} {method} seed={seed}, epochs={args.epochs}")
        return

    output = Path(args.output).resolve()
    old = read_json(output / "configuration.json")
    if output.exists() and not args.resume:
        parser.error("Output exists; use --resume or a new --output directory")
    if args.resume and not old:
        parser.error("--resume requires an existing configuration.json")
    from hsi_benchmark.data import download, prepare, resolve_revision
    # Resolve metadata before creating a fresh output directory. A connectivity
    # failure should not leave an empty output that the next invocation rejects.
    revision = old["hf_revision"] if old else resolve_revision(args.hf_revision)
    # The lock belongs to the separate output, so existing workflows are unaffected.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".launcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another launcher is using this output directory")
    config = {
        "tasks": tasks, "modes": modes, "models": args.models, "seeds": args.seeds,
        "epochs": args.epochs, "num_agents": args.num_agents, "num_workers": args.num_workers,
        "preprocessing": {"band_alignment": args.band_alignment, "fit_pixels": args.fit_pixels,
                          "chunk_pixels": args.chunk_pixels, "preprocess_seed": args.preprocess_seed},
        "requested_hf_revision": args.hf_revision, "hf_revision": revision,
        "data_dir": str(Path(args.data_dir).resolve()), "code_hash": code_hash(),
        "python": platform.python_version(), "packages": packages(),
        "evaluation_interval": "core/TSTnet/SSWADA every epoch; other recipes retain their schedule",
    }
    config["run_fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if old and old != config:
        parser.error("Configuration, code or package versions changed; use a new --output directory")
    atomic_json(output / "configuration.json", config)
    (output / "launcher_error.json").unlink(missing_ok=True)
    atomic_json(output / "preparation_errors.json", [])
    (output / ".gitignore").write_text("*\n")
    write_report(output, config)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    paths = code_paths()
    failures = []
    preparation_errors = []
    try:
        snapshot = snapshot_code(output, paths, config["code_hash"])
        if not args.prepare_only:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; activate the GPU PyTorch environment on the server")
            print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
        names = sorted({task[role] for task in tasks for role in ("source", "target")})
        location = download(args.data_dir, revision, names)
        for task in tasks:
            for mode in modes:
                print(f"Preparing {task['direction']} {mode}", flush=True)
                try:
                    manifest = prepare(location, output / "prepared", task, mode, config["preprocessing"])
                except Exception as error:
                    preparation_errors.append(f"{task['direction']}/{mode}: {error}")
                    atomic_json(output / "preparation_errors.json", preparation_errors)
                    for method in args.models:
                        for seed in args.seeds:
                            destination = output / "runs" / task["direction"] / mode / method / str(seed)
                            fingerprint = config["run_fingerprint"] + f"/{task['direction']}/{mode}/{method}/{seed}"
                            if args.resume and complete(destination, fingerprint, len(task["class_names"])):
                                continue
                            atomic_json(destination / "status.json", {"state": "failed", "error": "Preparation: " + str(error),
                                                                      "run_fingerprint": fingerprint})
                            (destination / "training.log").write_text("Preparation failed: " + str(error) + "\n")
                            failures.append(str(destination.relative_to(output)))
                    write_report(output, config)
                    continue
                write_report(output, config)
                if args.prepare_only:
                    continue
                for method in args.models:
                    for seed in args.seeds:
                        destination = output / "runs" / task["direction"] / mode / method / str(seed)
                        print(f"\nRUN {task['direction']} {mode} {method} seed={seed}", flush=True)
                        if not run_one(destination, task, mode, method, seed, manifest, config, args, paths, snapshot):
                            failures.append(str(destination.relative_to(output)))
                        atomic_json(output / "failures.json", failures)
                        write_report(output, config)
    except KeyboardInterrupt:
        write_report(output, config)
        print("Interrupted. Continue with the same options and --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        atomic_json(output / "launcher_error.json", {"error": str(error)})
        write_report(output, config)
        raise SystemExit(str(error))
    atomic_json(output / "failures.json", failures)
    write_report(output, config)
    print(f"Results: {output / 'results.md'}", flush=True)
    if failures or preparation_errors:
        raise SystemExit(f"{len(failures)} failed runs; {len(preparation_errors)} preparation failures. "
                         "See results.md and per-run training.log; retry with --resume.")


if __name__ == "__main__":
    main()
