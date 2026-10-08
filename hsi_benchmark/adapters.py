"""Process-local adapters for the existing recipes; no legacy files are edited.

The launcher supplies an isolated working tree. This module redirects data loading,
input/class dimensions and metric export, retaining each method's training loop.
AST edits are limited to named adapter points and fail closed when they disappear.
"""
import ast
import importlib
import json
import math
import os
from pathlib import Path
import shutil

from . import context, metrics
from .data import atomic_json


class PatchStore:
    """Extract patches into a disk-backed array instead of materializing them in RAM."""
    def __init__(self):
        self.counter = 0

    def extract(self, image, labels, halfwidth, indices=None, padding="constant"):
        import numpy as np
        folder = Path(os.environ["BIDA_RUN_DIR"]) / "patch_cache"
        folder.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        coordinates = np.argwhere(labels > 0) if indices is None else np.asarray(indices)
        if len(coordinates) == 0:
            raise ValueError("Empty patch dataset")
        size = 2 * halfwidth + 1
        patches = np.lib.format.open_memmap(
            folder / f"patches_{self.counter}.npy", mode="w+", dtype=np.float32,
            shape=(len(coordinates), image.shape[-1], size, size),
        )
        padded = np.pad(image, ((halfwidth, halfwidth), (halfwidth, halfwidth), (0, 0)), mode=padding)
        y = labels[coordinates[:, 0], coordinates[:, 1]].astype(np.int64)
        for index, (row, column) in enumerate(coordinates):
            patches[index] = padded[row:row + size, column:column + size].transpose(2, 0, 1)
        patches.flush()
        return patches, y

    def source(self, image, labels, halfwidth, count, padding="constant"):
        import numpy as np
        selected = []
        for class_id in range(1, int(labels.max()) + 1):
            coordinates = np.argwhere(labels == class_id)
            np.random.shuffle(coordinates)
            selected.extend(coordinates[:count])
        np.random.shuffle(selected)
        patches, y = self.extract(image, labels, halfwidth, selected, padding=padding)
        return patches, y - 1

    def target(self, image, labels, halfwidth, with_index=False, padding="constant"):
        import numpy as np
        coordinates = np.argwhere(labels > 0)
        permutation = np.random.permutation(len(coordinates))
        patches, y = self.extract(image, labels, halfwidth, coordinates[permutation], padding=padding)
        padded_gt = np.pad(labels, halfwidth, mode="constant")
        rows, columns = (coordinates + halfwidth).T
        values = (patches, y - 1, padded_gt, permutation, rows, columns)
        return (np.arange(len(y)),) + values if with_index else values


def install(method, info):
    """Patch imports before executing the corresponding legacy entry point."""
    import numpy as np
    import torch
    import restored_reporting
    store = PatchStore()
    captured = {}
    class_count = info["n_classes"]

    def from_confusion(cm, selection):
        if np.asarray(cm).shape != (class_count, class_count):
            raise ValueError("Legacy exporter returned an unexpected class dimension")
        score = metrics.from_confusion(cm, selection)
        captured["score"] = score
        return score

    def from_predictions(labels, predictions, selection):
        score = metrics.from_predictions(labels, predictions, selection, class_count)
        captured["score"] = score
        return score

    restored_reporting.from_confusion = from_confusion
    restored_reporting.from_predictions = from_predictions

    def core_load(name, directory, norm="normband"):
        return context.load_scene(context.role_for_name(name), zero_based=True)

    def external_load(image_file, label_file):
        image, labels, _ = context.load_scene(context.role_for_name(image_file))
        return image, labels

    def get_dataset(name, *args, **kwargs):
        image, labels, names = context.load_scene(context.role_for_name(name))
        return image, labels, names, [0], (0, 1, 2), None

    def bida_factory(module_name, class_name):
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        def factory(dataset, opts):
            args = dict(n_bands=info["n_bands"], num_classes=class_count,
                        num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth)
            if class_name == "AgentBiDAnet":
                args.update(heads=opts.num_heads, num_agents=opts.num_agents)
            return cls(**args)
        return factory

    if method in ("GAHT", "BiDA", "AgentBiDA", "SelfAttentionAgentBiDA"):
        dataset = importlib.import_module("utils.dataset")
        dataset.load_mat_hsi = core_load
        if method == "GAHT":
            model_module = importlib.import_module("models.GAHT")
            def gaht(dataset, patch_size):
                return model_module.MyTransformer(
                    img_size=patch_size, in_chans=info["n_bands"], num_classes=class_count,
                    n_groups=[4, 4, 4], depths=[1, 2, 1],
                )
            model_module.GAHT = gaht
            importlib.import_module("models.get_model").GAHT = gaht
        elif method == "BiDA":
            factory = bida_factory("models.BiDA", "BiDAnet")
            importlib.import_module("models.BiDA").BiDA = factory
            importlib.import_module("models.get_model").BiDA = factory
        else:
            module_name = ("models.agent_bida_fix" if method == "AgentBiDA"
                           else "models.self_attention_agent_bida_fix")
            importlib.import_module(module_name).AgentBiDA = bida_factory(module_name, "AgentBiDAnet")
    elif method == "TSTnet":
        importlib.import_module("datasets").get_dataset = get_dataset
    elif method == "SSWADA":
        importlib.import_module("datasets").get_dataset = get_dataset
        module = importlib.import_module("FeatureNet")
        original_gc = module.GC
        class AdaptiveGC(original_gc):
            def __init__(self, inplans, planes, stride=1):
                torch.nn.Module.__init__(self)
                self.conv = module.conv(inplans, planes, kernel_size=9, padding=4,
                                        stride=stride, groups=math.gcd(math.gcd(inplans, planes), 16))
                self.se = module.SEWeightModule(planes)
                self.split_channel = planes
                self.softmax = torch.nn.Softmax(dim=1)
        module.GC = AdaptiveGC
    elif method == "CLDA":
        module = importlib.import_module("datapre")
        module.load_data03 = external_load
        module.train_test_preclass = lambda *args: store.source(*args, padding="symmetric")
        module.all_data = lambda *args: store.target(*args, padding="symmetric")
    elif method in ("SCLUDA", "MLUDA"):
        config = importlib.import_module("config_Houston")
        config.nBand = info["n_bands"]
        config.CLASS_NUM = class_count
        if method == "MLUDA":
            # OpenCV guidedFilter accepts a one- or three-channel guide, not the
            # legacy Houston config's two-channel guide. This is internal guidance,
            # not the common input-band reduction under comparison.
            config.pca_n = 3
        module = importlib.import_module("utils")
        module.load_data_houston = external_load
        module.get_sample_data = store.source
        module.get_all_data = lambda *args: store.target(*args, with_index=True)
    elif method == "PCADA":
        module = importlib.import_module("datasets.data_pre")
        module.load_data_houston = external_load
        module.get_source_data = store.source
        module.get_target_data = store.target
        # PCADA's old exporter only has summary scalars. Capture the actual final
        # evaluation matrix so every new result is independently checkable.
        sklearn_metrics = importlib.import_module("sklearn.metrics")
        original = sklearn_metrics.confusion_matrix
        def confusion_matrix(labels, predictions, *args, **kwargs):
            kwargs["labels"] = list(range(class_count))
            cm = original(labels, predictions, *args, **kwargs)
            captured["pcada_cm"] = cm
            return cm
        sklearn_metrics.confusion_matrix = confusion_matrix
    elif method == "CACL":
        module = importlib.import_module("utils")
        module.single_data_loading = lambda *args: (*context.load_pair(), info["n_bands"])
        def patching(image, labels, patchsize=5):
            patches, y = store.extract(image, labels, patchsize // 2, padding="symmetric")
            return torch.from_numpy(patches), torch.from_numpy(y - 1)
        module.single_data_patching = patching
    elif method == "MDGTnet":
        cls = importlib.import_module("networks.MDGTnet").MDGTnet
        original_forward = cls.forward
        def forward(model, intra, inter):
            logits, *features = original_forward(model, intra, inter)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            return (logits, *features)
        cls.forward = forward
    else:
        raise ValueError("Unsupported method: " + method)
    return store, captured


class RecipeAdapter(ast.NodeTransformer):
    """Adjust only the standalone entry-point glue, never loss/training expressions."""
    TENSOR_INPUTS = {"trainX", "testX", "train_xs", "train_data_s", "test_data"}

    def __init__(self, method, info):
        self.method = method
        self.info = info
        self.changed = set()

    def visit_FunctionDef(self, node):
        if self.method == "MDGTnet" and node.name == "load_mat_h5py":
            node.body = ast.parse("return hsi_external_load(img_path, lbl_path)").body
            self.changed.add("load_mat_h5py")
        elif self.method == "MDGTnet" and node.name == "extract_patches":
            node.body = ast.parse("return hsi_mdgt_patches(img, label, patch_size)").body
            self.changed.add("extract_patches")
        return self.generic_visit(node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (self.method == "MDGTnet" and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "spec_range"):
            # [0, C] creates an empty first spectral branch in the old adapter.
            # Both branches need positive widths; partition the input at its midpoint.
            node.value = ast.List([ast.Constant(self.info["n_bands"] // 2),
                                   ast.Constant(self.info["n_bands"])], ast.Load())
            self.changed.add("spec_range")
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant):
            name = node.targets[0].id
            constants = {"nBand": self.info["n_bands"], "CLASS_NUM": self.info["n_classes"],
                         "NUM_CLASSES": self.info["n_classes"]}
            if name in constants:
                node.value = ast.Constant(constants[name])
                self.changed.add(name)
            elif name == "source_name" and node.value.value == "Houston13":
                node.value = ast.Constant(self.info["source_name"])
            elif name == "target_name" and node.value.value == "Houston18":
                node.value = ast.Constant(self.info["target_name"])
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "parse_args"):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Unexpected argument-parser assignment")
            self.changed.add("parse_args")
            configure = ast.Expr(ast.Call(ast.Name("hsi_configure_args", ast.Load()),
                                         [ast.Name(node.targets[0].id, ast.Load())], []))
            # Core parsers live inside an if __name__ block; external parsers are
            # module-level. NodeTransformer statement expansion handles both.
            return [node, ast.copy_location(configure, node)]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        # Share disk-backed data with PyTorch rather than torch.tensor's full copy.
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch" and node.func.attr == "tensor"
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Name) and node.args[0].id in self.TENSOR_INPUTS):
            return ast.copy_location(ast.Call(ast.Name("hsi_patch_tensor", ast.Load()), node.args, []), node)
        return node


def execute(method, entry):
    import numpy as np
    import torch
    info = context.manifest()
    store, captured = install(method, info)
    def configure_args(args):
        args.n_bands = info["n_bands"]
        args.n_classes = info["n_classes"]
        if hasattr(args, "source_name"):
            args.source_name = info["source_name"]
            args.target_name = info["target_name"]
        if method == "CACL":
            args.in_channel = info["n_bands"]
            args.num_classes = info["n_classes"]
    def external_load(image_file, label_file):
        image, labels, _ = context.load_scene(context.role_for_name(image_file))
        return image, labels
    def mdgt_patches(image, labels, size):
        return store.extract(image.transpose(1, 2, 0), labels, size // 2, padding="reflect")

    entry = Path(entry)
    tree = ast.parse(entry.read_text(), filename=str(entry))
    adapter = RecipeAdapter(method, info)
    tree = adapter.visit(tree)
    required = {"parse_args"}
    if method in ("CLDA", "PCADA"):
        required.update(("nBand", "CLASS_NUM"))
    elif method == "MDGTnet":
        required.update(("NUM_CLASSES", "load_mat_h5py", "extract_patches", "spec_range"))
    if not required.issubset(adapter.changed):
        raise ValueError("Legacy adapter points changed: " + str(required - adapter.changed))
    ast.fix_missing_locations(tree)
    namespace = {
        "__name__": "__main__", "__file__": str(entry), "__package__": None,
        "hsi_configure_args": configure_args, "hsi_external_load": external_load,
        "hsi_mdgt_patches": mdgt_patches,
        "hsi_patch_tensor": lambda array: torch.from_numpy(np.asarray(array)),
    }
    print(f"Adapter: {method}, {info['source_name']} -> {info['target_name']}, "
          f"{info['mode']}, {info['n_bands']} input bands, {info['n_classes']} classes", flush=True)
    exec(compile(tree, str(entry), "exec"), namespace)
    if method == "PCADA":
        scores = metrics.from_confusion(captured["pcada_cm"], "final_epoch")
    elif method == "MDGTnet":
        scores = metrics.from_predictions(namespace["all_gts"], namespace["all_preds"],
                                          "final_epoch", info["n_classes"])
    else:
        scores = captured.get("score") or namespace.get("res_dict")
        if scores is None:
            # Recovery can finish without another evaluation when the final epoch
            # was already saved; the core loop re-exports its saved score instead.
            relative = "ablation_results" if method in ("GAHT", "BiDA", "AgentBiDA", "SelfAttentionAgentBiDA") else "../../ablation_results"
            seed = Path(os.environ["BIDA_RUN_DIR"]).name
            path = Path(relative) / f"{method}_results_seed_{seed}.json"
            if path.exists():
                scores = json.loads(path.read_text())
        if scores is None:
            raise ValueError("The method did not export evaluated predictions")
    metrics.validate(scores, info["n_classes"])
    scores.update(source=info["source_name"], target=info["target_name"],
                  mode=info["mode"], input_bands=info["n_bands"],
                  class_names=info["class_names"], run_fingerprint=os.environ["BIDA_RUN_FINGERPRINT"])
    if method == "SSWADA":
        scores["spectral_convolution_groups"] = math.gcd(info["n_bands"], 16)
    if method == "MLUDA":
        scores["internal_guidance_pca_components"] = 3
    if method == "MDGTnet":
        scores["spectral_partition"] = [info["n_bands"] // 2, info["n_bands"]]
    destination = Path(os.environ["BIDA_RUN_DIR"])
    atomic_json(destination / "result.json", scores)
    # Keep final/best legacy weight files when they are present. Core and CACL
    # checkpoints already respect BIDA_RUN_DIR.
    candidates = []
    if method == "TSTnet" and "model_save_path" in namespace:
        candidates = [Path(namespace["model_save_path"])]
    elif method == "PCADA":
        candidates = sorted(Path("checkpoints/houston").glob("*.pth"))
    elif method == "MDGTnet":
        candidates = [Path("models/MDGTnet_H1318") / f"model{namespace['epoch']}.pth"]
    elif method == "SSWADA":
        candidates = sorted(Path("Result_comparison").rglob("*.pth"))
    for path in candidates:
        if path.is_file():
            folder = destination / "checkpoints"
            folder.mkdir(exist_ok=True)
            shutil.copy2(path, folder / path.name)
