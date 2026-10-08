"""Offline checks for the new workflow; run on the server, not automatically.

python -m unittest discover -s tests -p 'test_dataset_benchmark.py'
No real dataset download, training, CUDA allocation or legacy entry execution.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import scipy.io

from hsi_benchmark.adapters import PatchStore, RecipeAdapter
from hsi_benchmark.catalog import DATASET_FOLDERS, METHODS, PAIRS, directions
from hsi_benchmark.context import load_scene as load_prepared_scene, role_for_name
from hsi_benchmark.data import atomic_json, load_scene, prepare, remap_labels
from hsi_benchmark.metrics import from_confusion, from_predictions, validate
from hsi_benchmark.reporting import write_report
from run_dataset_benchmark import ENTRIES, ROOT, complete, method_arguments, snapshot_code, workspace


class CatalogTests(unittest.TestCase):
    def test_hyrank_raw_ids_preserve_absent_class_gaps(self):
        shared = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14]
        scenes = {"Loukia": np.arange(15).reshape(1, -1),
                  "Dioni": np.array([0] + shared).reshape(1, -1)}
        for task in directions(["Loukia-Dioni"]):
            self.assertEqual(task["source_ids"], shared)
            self.assertEqual(task["target_ids"], shared)
            for role in ("source", "target"):
                raw = scenes[task[role]]
                mapped = remap_labels(raw, task[role + "_ids"], task[role])
                np.testing.assert_array_equal(mapped[raw == 7], [6])
                np.testing.assert_array_equal(mapped[raw == 9], [7])
                np.testing.assert_array_equal(mapped[raw == 14], [12])
                self.assertTrue(np.all(mapped[np.isin(raw, [0, 6, 8])] == 0))
        with self.assertRaisesRegex(ValueError, "in Dioni.*Available finite GT label counts"):
            remap_labels(scenes["Dioni"], list(range(1, 13)), "Dioni")

    def test_eight_directions_and_semantic_reverse(self):
        tasks = list(directions(list(PAIRS)))
        self.assertEqual(len(tasks), 8)
        for forward, reverse in zip(tasks[::2], tasks[1::2]):
            self.assertEqual(forward["source"], reverse["target"])
            self.assertEqual(forward["source_ids"], reverse["target_ids"])
            self.assertEqual(forward["class_names"], reverse["class_names"])
        mars = next(task for task in tasks if task["source"] == "Utopia")
        self.assertEqual(mars["source_ids"], [2, 4, 8, 9])
        self.assertEqual(mars["target_ids"], [4, 5, 7, 8])

    def test_background_and_scene_specific_classes_are_ignored(self):
        raw = np.array([[0, 1, 2, 3], [4, 8, 9, 7]])
        mapped = remap_labels(raw, [2, 4, 8, 9])
        np.testing.assert_array_equal(mapped, [[0, 0, 1, 0], [2, 3, 4, 0]])
        with self.assertRaises(ValueError):
            remap_labels(raw, [10])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.task = next(directions(["PaviaU-PaviaC"], "forward"))
        # A small two-class fixture; spectra are not tied to labels.
        self.task["class_names"] = ["A", "B"]
        self.task["source_ids"] = [1, 2]
        self.task["target_ids"] = [3, 4]
        rng = np.random.default_rng(12)
        self.source = rng.normal(size=(8, 9, 32)).astype(np.float32)
        self.target = rng.normal(1, 2, size=(7, 9, 31)).astype(np.float32)
        self.source_gt = np.tile(np.arange(9) % 2 + 1, (8, 1)).astype(np.int32)
        self.target_gt = np.tile(np.arange(9) % 2 + 3, (7, 1)).astype(np.int32)
        self.config = {"band_alignment": "resample", "fit_pixels": 50,
                       "chunk_pixels": 11, "preprocess_seed": 0}
        self.write_scene("PaviaU", self.source, self.source_gt)
        self.write_scene("PaviaC", self.target, self.target_gt)

    def tearDown(self):
        self.directory.cleanup()

    def write_scene(self, name, image, labels, hdf5=False):
        folder = self.root / "raw" / DATASET_FOLDERS[name]
        folder.mkdir(parents=True, exist_ok=True)
        if hdf5:
            for suffix, array in (("data", image), ("gt", labels)):
                with h5py.File(folder / f"scene_{suffix}.mat", "w") as stream:
                    dataset = stream.create_dataset("arbitrary_key", data=array.T)
                    dataset.attrs["MATLAB_class"] = np.bytes_("double")
        else:
            scipy.io.savemat(folder / "scene_data.mat", {"arbitrary_cube_key": image})
            scipy.io.savemat(folder / "scene_gt.mat", {"arbitrary_gt_key": labels})

    def test_matlab_v73_and_scipy_have_same_orientation(self):
        image, labels, _, _ = load_scene(self.root / "raw", "PaviaU")
        np.testing.assert_array_equal(image, self.source)
        np.testing.assert_array_equal(labels, self.source_gt)
        self.write_scene("PaviaU", self.source, self.source_gt, hdf5=True)
        image, labels, _, _ = load_scene(self.root / "raw", "PaviaU")
        np.testing.assert_array_equal(image, self.source)
        np.testing.assert_array_equal(labels, self.source_gt)

    def test_pca30_same_basis_and_target_labels_do_not_fit_transforms(self):
        manifest_path = prepare(self.root / "raw", self.root / "prepared", self.task, "pca30", self.config)
        info = json.loads(manifest_path.read_text())
        self.assertEqual(info["n_bands"], 30)
        self.assertEqual(np.load(info["source"]["image"]).shape, (8, 9, 30))
        self.assertEqual(np.load(info["target"]["image"]).shape, (7, 9, 30))
        with np.load(manifest_path.parent / "transform.npz") as transform:
            target = self.target.reshape(-1, 31)
            standardized = (target - transform["scaler_mean"]) / transform["scaler_scale"]
            expected = (standardized - transform["pca_mean"]) @ transform["pca_components"].T
            np.testing.assert_allclose(np.load(info["target"]["image"]).reshape(-1, 30), expected,
                                       rtol=2e-5, atol=2e-5)
            original_components = transform["pca_components"].copy()
        self.write_scene("PaviaC", self.target, np.where(self.target_gt == 3, 4, 3))
        second = prepare(self.root / "raw", self.root / "other", self.task, "pca30", self.config)
        with np.load(second.parent / "transform.npz") as transform:
            np.testing.assert_array_equal(original_components, transform["pca_components"])

    def test_no_pca_band_count_cache_reuse_and_tamper_detection(self):
        path = prepare(self.root / "raw", self.root / "prepared", self.task, "no_pca", self.config)
        info = json.loads(path.read_text())
        self.assertEqual(info["n_bands"], 31)
        self.assertIsNone(info["pca_n_components"])
        self.assertEqual(path, prepare(self.root / "raw", self.root / "prepared", self.task, "no_pca", self.config))
        with Path(info["target"]["labels"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            prepare(self.root / "raw", self.root / "prepared", self.task, "no_pca", self.config)

    def test_strict_bands_and_nonfinite_mask(self):
        with self.assertRaises(ValueError):
            prepare(self.root / "raw", self.root / "prepared", self.task, "pca30",
                    {**self.config, "band_alignment": "error"})
        self.source[0, 0, 0] = np.nan
        self.write_scene("PaviaU", self.source, self.source_gt)
        image, labels, valid, info = load_scene(self.root / "raw", "PaviaU")
        self.assertFalse(valid[0, 0])
        self.assertEqual(labels[0, 0], 0)
        self.assertTrue(np.isfinite(image).all())
        self.assertEqual(info["invalid_spectra"], 1)

    def test_context_role_aliases_and_label_conventions(self):
        path = prepare(self.root / "raw", self.root / "prepared", self.task, "pca30", self.config)
        with patch.dict(os.environ, {"HSI_BENCHMARK_MANIFEST": str(path)}):
            self.assertEqual(role_for_name("../../Houston/Houston13.mat"), "source")
            self.assertEqual(role_for_name("PaviaC"), "target")
            _, one_based, _ = load_prepared_scene("target")
            _, zero_based, _ = load_prepared_scene("target", zero_based=True)
            np.testing.assert_array_equal(one_based - 1, zero_based)


class AdapterTests(unittest.TestCase):
    def test_workers_use_frozen_sources_and_modified_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source_root = folder / "checkout"
            source_root.mkdir()
            source = source_root / "model.py"
            source.write_text("version = 1\n")
            with patch("run_dataset_benchmark.ROOT", source_root):
                snapshot = snapshot_code(folder / "output", [source])
                work = workspace(folder / "run", [source], snapshot)
                source.write_text("version = 2\n")
                self.assertEqual((work / "model.py").read_text(), "version = 1\n")
                with self.assertRaises(ValueError):
                    snapshot_code(folder / "output", [source])

    def test_all_actual_entry_points_have_required_adapter_points(self):
        info = {"n_bands": 30, "n_classes": 4, "source_name": "Utopia", "target_name": "NiliFossae"}
        for method in METHODS:
            with self.subTest(method=method):
                path = ROOT / ENTRIES[method]
                adapter = RecipeAdapter(method, info)
                tree = adapter.visit(ast.parse(path.read_text(), filename=str(path)))
                ast.fix_missing_locations(tree)
                compile(tree, str(path), "exec")  # compile only; never run the entry
                self.assertIn("parse_args", adapter.changed)
                if method in ("CLDA", "PCADA"):
                    self.assertTrue({"nBand", "CLASS_NUM"}.issubset(adapter.changed))
                if method == "MDGTnet":
                    self.assertTrue({"NUM_CLASSES", "load_mat_h5py", "extract_patches", "spec_range"}.issubset(adapter.changed))
                arguments = method_arguments(method, "Utopia", "NiliFossae", 2100, 1, 4, 0)
                self.assertIn("--seed", arguments)

    def test_disk_backed_patch_centers_and_target_coordinate_mapping(self):
        image = np.arange(5 * 6 * 3, dtype=np.float32).reshape(5, 6, 3)
        labels = np.zeros((5, 6), dtype=np.int64)
        labels[0, 0], labels[2, 3], labels[4, 5] = 1, 2, 1
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"BIDA_RUN_DIR": directory}):
            patches, y, gt, permutation, rows, columns = PatchStore().target(image, labels, 1)
            self.assertIsInstance(patches, np.memmap)
            for index, original in enumerate(permutation):
                row, column = rows[original], columns[original]
                np.testing.assert_array_equal(patches[index, :, 1, 1], image[row - 1, column - 1])
                self.assertEqual(y[index], gt[row, column] - 1)


class MetricsAndReportingTests(unittest.TestCase):
    def test_metrics_use_actual_counts_and_arbitrary_class_count(self):
        for class_count in (4, 6, 7, 12):
            labels = np.repeat(np.arange(class_count), 3)
            score = from_predictions(labels, labels, "final_epoch", class_count)
            validate(score, class_count)
            self.assertEqual(score["OA"], 100)
            self.assertEqual(score["Kappa"], 100)
            self.assertEqual(len(score["classes"]), class_count)
        with self.assertRaises(ValueError):
            from_confusion([[3, 0], [0, 0]], "final_epoch")

    def test_single_seed_scores_and_stale_scores_are_rejected(self):
        task = next(directions(["Utopia-NiliFossae"], "forward"))
        config = {"tasks": [task], "modes": ["pca30"], "models": ["GAHT"], "seeds": [1],
                  "epochs": 1, "run_fingerprint": "fixture", "hf_revision": "fixture",
                  "preprocessing": {"band_alignment": "resample"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, config)
            self.assertIn("| GAHT | 0/1 | pending | pending | pending |", (root / "results.md").read_text())
            destination = root / "runs" / task["direction"] / "pca30" / "GAHT" / "1"
            fingerprint = "fixture/" + task["direction"] + "/pca30/GAHT/1"
            score = from_confusion(np.eye(4, dtype=int), "final_epoch")
            score["run_fingerprint"] = fingerprint
            atomic_json(destination / "result.json", score)
            atomic_json(destination / "status.json", {"state": "complete", "run_fingerprint": fingerprint})
            self.assertTrue(complete(destination, fingerprint, 4))
            self.assertFalse(complete(destination, "different", 4))
            write_report(root, config)
            report = (root / "results.md").read_text()
            self.assertIn("| GAHT | 1/1 | 100.00 | 100.00 | 100.00 |", report)
            self.assertNotIn("±", report)
            score["run_fingerprint"] = "different"
            atomic_json(destination / "result.json", score)
            write_report(root, config)
            self.assertIn("| GAHT | 0/1 | pending | pending | pending |", (root / "results.md").read_text())
            config["seeds"] = [1, 2]
            with self.assertRaisesRegex(ValueError, "exactly one training seed"):
                write_report(root, config)


if __name__ == "__main__":
    unittest.main()
