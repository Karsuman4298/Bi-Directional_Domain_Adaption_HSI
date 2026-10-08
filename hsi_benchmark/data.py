"""Hub downloads, MAT loading, semantic remapping and shared preprocessing.

Imported only when the remote launcher prepares data. No download occurs at import.
"""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

from .catalog import DATASET_FOLDERS, REPO_ID


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_revision(revision):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(REPO_ID, revision=revision).sha


def download(data_dir, revision, dataset_names):
    from huggingface_hub import snapshot_download
    # Pin the resolved commit, restrict downloads to the selected scenes, and keep
    # the Hub's own incremental download metadata for interrupted transfers.
    location = Path(data_dir).resolve() / revision
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=revision,
        local_dir=str(location),
        allow_patterns=[DATASET_FOLDERS[name] + "/*.mat" for name in dataset_names],
    )
    return location


def find_mat_files(location, dataset_name):
    folder = Path(location) / DATASET_FOLDERS[dataset_name]
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".mat")
    def is_gt(path):
        stem = path.stem.lower()
        return stem == "gt" or stem.endswith("_gt") or "groundtruth" in stem or "ground_truth" in stem
    labels = [path for path in files if is_gt(path)]
    cubes = [path for path in files if not is_gt(path)]
    if len(labels) != 1 or len(cubes) != 1:
        raise ValueError(f"Expected one cube and one GT file in {folder}; found {files}")
    return cubes[0], labels[0]


def read_array(path, ndim):
    import numpy as np
    import scipy.io
    import h5py
    # HDF5 detection also avoids depending on which error scipy emits for v7.3.
    if h5py.is_hdf5(path):
        arrays = {}
        with h5py.File(path, "r") as stream:
            for key, item in stream.items():
                if isinstance(item, h5py.Dataset) and item.ndim == ndim and item.dtype.kind in "biuf":
                    value = np.asarray(item)
                    if "MATLAB_class" in item.attrs:
                        value = value.transpose(tuple(reversed(range(value.ndim))))
                    arrays[key] = value
    else:
        arrays = {key: value for key, value in scipy.io.loadmat(path).items()
                  if not key.startswith("__") and isinstance(value, np.ndarray)
                  and value.ndim == ndim and value.dtype.kind in "biuf"}
    if len(arrays) != 1:
        raise ValueError(f"Expected one numeric {ndim}D array in {path}; candidates: {list(arrays)}")
    key, array = next(iter(arrays.items()))
    return array, key


def load_scene(location, dataset_name):
    import numpy as np
    cube_file, gt_file = find_mat_files(location, dataset_name)
    cube, cube_key = read_array(cube_file, 3)
    gt, gt_key = read_array(gt_file, 2)
    if not np.isfinite(gt).all() or (gt < 0).any() or (gt != np.floor(gt)).any():
        raise ValueError(f"Invalid GT IDs in {gt_file}")
    axes = [axes for axes in itertools.permutations(range(3))
            if (cube.shape[axes[0]], cube.shape[axes[1]]) == gt.shape]
    if len(axes) != 1:
        raise ValueError(f"Cannot unambiguously align cube {cube.shape} with GT {gt.shape} in {dataset_name}")
    cube = np.asarray(cube.transpose(axes[0]), dtype=np.float32, order="C")
    gt = gt.astype(np.int32)
    # Invalid spectra become background for training and evaluation. Use the same
    # finite mask for both modes, before fitting any unsupervised transforms.
    valid = np.isfinite(cube).all(axis=2)
    invalid_count = int((~valid).sum())
    if invalid_count:
        gt[~valid] = 0
        cube[~valid] = 0
    return cube, gt, valid, {
        "cube_file": str(cube_file), "gt_file": str(gt_file),
        "cube_key": cube_key, "gt_key": gt_key,
        "cube_sha256": sha256_file(cube_file), "gt_sha256": sha256_file(gt_file),
        "shape": list(cube.shape), "original_ids": np.unique(gt).tolist(),
        "invalid_spectra": invalid_count,
    }


def remap_labels(labels, original_ids):
    import numpy as np
    output = np.zeros(labels.shape, dtype=np.int32)
    for class_id, original_id in enumerate(original_ids, 1):
        mask = labels == original_id
        if not mask.any():
            raise ValueError(f"Shared original class ID {original_id} has no finite labeled pixels")
        output[mask] = class_id
    return output


def align_bands(spectra, bands, policy):
    import numpy as np
    if spectra.shape[1] == bands:
        return np.asarray(spectra, dtype=np.float32)
    if policy == "error":
        raise ValueError("Unequal native band counts; select --band-alignment resample or truncate")
    if policy == "truncate":
        return np.asarray(spectra[:, :bands], dtype=np.float32)
    # Linear interpolation on normalized band index; no claim of physical
    # wavelength alignment is made when wavelength metadata is unavailable.
    positions = np.linspace(0, spectra.shape[1] - 1, bands)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    fraction = (positions - lower).astype(np.float32)
    return spectra[:, lower] * (1 - fraction) + spectra[:, upper] * fraction


def prepare(location, cache_root, task, mode, config):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    folder = Path(cache_root).resolve() / task["direction"] / mode
    expected = {"task": task, "mode": mode, "preprocessing": config,
                "location": str(Path(location).resolve())}
    old_path = folder / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if old.get("cache_configuration") != expected:
            raise ValueError(f"Preprocessing cache differs at {folder}; use a new output directory")
        for role in ("source", "target"):
            for kind in ("image", "labels"):
                path = Path(old[role][kind])
                if sha256_file(path) != old[role][kind + "_sha256"]:
                    raise ValueError(f"Prepared cache file changed: {path}")
            raw = old[role]["raw"]
            for kind in ("cube", "gt"):
                if sha256_file(raw[kind + "_file"]) != raw[kind + "_sha256"]:
                    raise ValueError(f"Downloaded MAT file changed: {raw[kind + '_file']}")
        return old_path

    source, source_gt, source_valid, source_meta = load_scene(location, task["source"])
    target, target_gt, target_valid, target_meta = load_scene(location, task["target"])
    source_gt = remap_labels(source_gt, task["source_ids"])
    target_gt = remap_labels(target_gt, task["target_ids"])
    native_bands = [source.shape[-1], target.shape[-1]]
    bands = min(native_bands)
    if native_bands[0] != native_bands[1] and config["band_alignment"] == "error":
        raise ValueError(f"{task['direction']} has unequal band counts {native_bands}")
    source_flat = source.reshape(-1, source.shape[-1])
    fit_indices = np.flatnonzero(source_valid.reshape(-1))
    rng = np.random.default_rng(config["preprocess_seed"])
    if len(fit_indices) > config["fit_pixels"]:
        fit_indices = rng.choice(fit_indices, config["fit_pixels"], replace=False)
    fit = align_bands(source_flat[fit_indices], bands, config["band_alignment"])
    if len(fit) < 30 or bands < 30:
        raise ValueError("Need at least 30 source spectra and 30 aligned input bands for this suite")
    # Source-only fit; target GT never contributes to normalization or PCA.
    scaler = StandardScaler().fit(fit)
    fit = scaler.transform(fit).astype(np.float32)
    pca = None
    if mode == "pca30":
        pca = PCA(n_components=30, svd_solver="randomized", random_state=config["preprocess_seed"])
        pca.fit(fit)
    output_bands = 30 if pca is not None else bands
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=mode + ".", dir=folder.parent))
    info = {
        "cache_configuration": expected, "source_name": task["source"],
        "target_name": task["target"], "mode": mode, "n_bands": output_bands,
        "n_classes": len(task["class_names"]), "class_names": task["class_names"],
        "source_original_ids": task["source_ids"], "target_original_ids": task["target_ids"],
        "assumptions": task["assumptions"], "native_bands": native_bands,
        "aligned_bands": bands, "band_alignment": config["band_alignment"],
        "fit_domain": "source finite spectra without labels", "fit_pixels": len(fit_indices),
        "preprocess_seed": config["preprocess_seed"], "pca_n_components": 30 if pca else None,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist() if pca else None,
    }
    try:
        for role, image, labels, valid, raw in (
            ("source", source, source_gt, source_valid, source_meta),
            ("target", target, target_gt, target_valid, target_meta),
        ):
            image_path = temporary / (role + "_image.npy")
            labels_path = temporary / (role + "_labels.npy")
            output = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float32,
                                               shape=(*image.shape[:2], output_bands))
            output_flat = output.reshape(-1, output_bands)
            flat = image.reshape(-1, image.shape[-1])
            valid_flat = valid.reshape(-1)
            for start in range(0, len(flat), config["chunk_pixels"]):
                end = min(start + config["chunk_pixels"], len(flat))
                block = align_bands(flat[start:end], bands, config["band_alignment"])
                block = scaler.transform(block).astype(np.float32)
                if pca is not None:
                    block = pca.transform(block).astype(np.float32)
                block[~valid_flat[start:end]] = 0
                if not np.isfinite(block).all():
                    raise ValueError(f"Preprocessing produced nonfinite spectra in {role}")
                output_flat[start:end] = block
            output.flush()
            del output_flat, output
            np.save(labels_path, labels)
            info[role] = {
                "image": str(folder / image_path.name), "labels": str(folder / labels_path.name),
                "image_sha256": sha256_file(image_path), "labels_sha256": sha256_file(labels_path),
                "raw": raw, "support": np.bincount(labels.reshape(-1), minlength=info["n_classes"] + 1)[1:].tolist(),
            }
        transform = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                     "fit_indices": fit_indices}
        if pca is not None:
            transform.update(pca_components=pca.components_, pca_mean=pca.mean_,
                             explained_variance_ratio=pca.explained_variance_ratio_)
        np.savez(temporary / "transform.npz", **transform)
        atomic_json(temporary / "manifest.json", info)
        temporary.replace(folder)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return folder / "manifest.json"
