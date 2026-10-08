"""Internal remote-worker entry point; normally invoked by run_dataset_benchmark.py."""
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 3 or not os.environ.get("HSI_BENCHMARK_MANIFEST"):
        raise SystemExit("Use run_dataset_benchmark.py or one of the new Bash scripts")
    method, entry, *arguments = sys.argv[1:]
    # Local external utils/models must precede the repository's namesakes.
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = [entry, *arguments]
    from hsi_benchmark.adapters import execute
    execute(method, entry)


if __name__ == "__main__":
    main()
