"""Prepared data consumed by the separate process-local legacy adapters."""
import json
import os
from pathlib import Path


def manifest():
    path = os.environ.get("HSI_BENCHMARK_MANIFEST")
    if not path:
        raise RuntimeError("This adapter requires HSI_BENCHMARK_MANIFEST")
    return json.loads(Path(path).read_text())


def load_scene(role, zero_based=False):
    import numpy as np
    info = manifest()
    scene = info[role]
    # Read-only cubes, private writable label arrays. Legacy augmentation operates
    # on extracted patches, not on the shared preprocessing cache.
    image = np.load(scene["image"], mmap_mode="r")
    labels = np.load(scene["labels"]).astype(np.int64)
    if zero_based:
        labels -= 1
    return image, labels, list(info["class_names"])


def role_for_name(name):
    info = manifest()
    stem = Path(str(name)).stem
    # Forward Houston entry points are reused in an isolated process. These aliases
    # refer to source/target roles even when the real experiment is reversed.
    if stem == info["source_name"] or stem == "Houston13":
        return "source"
    if stem == info["target_name"] or stem == "Houston18":
        return "target"
    raise ValueError("Unexpected scene requested by legacy method: " + str(name))


def load_pair():
    source, source_gt, _ = load_scene("source")
    target, target_gt, _ = load_scene("target")
    return source, source_gt, target, target_gt
