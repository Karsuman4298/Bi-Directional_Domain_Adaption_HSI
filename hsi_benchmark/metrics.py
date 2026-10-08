"""Class-count-independent metrics, with percentages and Kappa x100."""
import math


def from_confusion(cm, selection):
    import numpy as np
    matrix = np.asarray(cm, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            matrix.shape[0] < 2 or not np.isfinite(matrix).all() or
            (matrix < 0).any() or (matrix != np.floor(matrix)).any()):
        raise ValueError("Expected a square finite nonnegative integer confusion matrix")
    support = matrix.sum(axis=1)
    if (support == 0).any():
        raise ValueError("An evaluation class is absent; cannot report closed-set AA")
    count = float(matrix.sum())
    recalls = matrix.diagonal() / support
    oa = float(matrix.trace() / count)
    pe = float((support * matrix.sum(axis=0)).sum() / count**2)
    if pe >= 1:
        raise ValueError("Kappa is undefined for this evaluation population")
    return {
        "OA": 100 * oa, "AA": float(100 * recalls.mean()),
        "Kappa": float(100 * (oa - pe) / (1 - pe)),
        "classes": {str(i + 1): float(100 * value) for i, value in enumerate(recalls)},
        "support": support.astype(int).tolist(),
        "confusion_matrix": matrix.astype(int).tolist(), "selection": selection,
    }


def from_predictions(labels, predictions, selection, class_count):
    import numpy as np
    labels, predictions = [np.asarray(value).reshape(-1) for value in (labels, predictions)]
    if labels.shape != predictions.shape or not len(labels):
        raise ValueError("Prediction/label count mismatch or empty evaluation")
    for value in (labels, predictions):
        if (not np.isfinite(value).all() or (value != np.floor(value)).any() or
                (value < 0).any() or (value >= class_count).any()):
            raise ValueError("Expected contiguous zero-based class IDs")
    cm = np.bincount(class_count * labels.astype(int) + predictions.astype(int),
                     minlength=class_count**2).reshape(class_count, class_count)
    return from_confusion(cm, selection)


def validate(scores, class_count):
    expected = from_confusion(scores["confusion_matrix"], scores["selection"])
    if len(expected["support"]) != class_count or scores["support"] != expected["support"]:
        raise ValueError("Class count or support differs from the confusion matrix")
    if set(scores["classes"]) != set(expected["classes"]):
        raise ValueError("Incorrect per-class keys")
    values = [(scores[key], expected[key]) for key in ("OA", "AA", "Kappa")]
    values += [(scores["classes"][key], value) for key, value in expected["classes"].items()]
    if any(not math.isfinite(float(value)) or abs(float(value) - target) > 1e-6
           for value, target in values):
        raise ValueError("Scores disagree with the confusion matrix")
