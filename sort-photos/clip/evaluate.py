"""Evaluation on held-out photos: metrics, confusion matrix, misclassifications and a Markdown report."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix  # noqa: E402

from .config import ALBUMS  # noqa: E402
from .features import Skipped, open_image  # noqa: E402
from .predict import PhotoClassifier, Prediction  # noqa: E402
from .train import collect_labeled_images  # noqa: E402


@dataclass
class EvaluationResult:
    """Everything computed by :func:`evaluate`."""

    class_names: List[str]
    predictions: List[Prediction]
    true_labels: List[str]
    accuracy: float
    per_class: Dict[str, Dict[str, float]]
    confusion: np.ndarray
    skipped: List[Skipped]

    @property
    def n_correct(self) -> int:
        return sum(p.label == t for p, t in zip(self.predictions, self.true_labels))

    def misclassified(self) -> List[Dict[str, Any]]:
        """Wrong predictions, most confidently wrong first."""
        rows = [self._row(p, t) for p, t in zip(self.predictions, self.true_labels) if p.label != t]
        return sorted(rows, key=lambda r: r["confidence"], reverse=True)

    def least_confident(self, k: int = 5) -> List[Dict[str, Any]]:
        """The ``k`` predictions with the lowest probability on the true class."""
        rows = [self._row(p, t) for p, t in zip(self.predictions, self.true_labels)]
        return sorted(rows, key=lambda r: r["probabilities"][r["true"]])[:k]

    def difficult_example(self) -> Dict[str, Any]:
        """The image with the lowest probability on its true class (a misclassification if any exist)."""
        return self.least_confident(1)[0]

    @staticmethod
    def _row(p: Prediction, true: str) -> Dict[str, Any]:
        return {"path": p.path, "true": true, "predicted": p.label, "confidence": p.confidence,
                "probabilities": p.probabilities}


def evaluate(classifier: PhotoClassifier, data_dir: Path, batch_size: int, progress: bool = True) -> EvaluationResult:
    """Classify every image in ``data_dir/<class>/`` and compare with the folder label.

    Raises
    ------
    ValueError
        If none of the images can be decoded.
    """
    paths, labels = collect_labeled_images(data_dir, classifier.class_names)
    truth = {path: classifier.class_names[i] for path, i in zip(paths, labels)}
    predictions, skipped = classifier.predict_paths(paths, batch_size, progress, desc="Evaluating")
    if not predictions:
        raise ValueError(f"No readable test images in {data_dir} ({len(skipped)} unreadable files skipped).")
    y_true = [truth[p.path] for p in predictions]
    y_pred = [p.label for p in predictions]
    names = classifier.class_names
    report = classification_report(y_true, y_pred, labels=names, output_dict=True, zero_division=0)
    return EvaluationResult(
        class_names=names,
        predictions=predictions,
        true_labels=y_true,
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class={n: report[n] for n in names},
        confusion=confusion_matrix(y_true, y_pred, labels=names),
        skipped=skipped,
    )


def _rel(path: Path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


def _probs(row: Dict[str, Any]) -> str:
    return ", ".join(f"{name} {p:.2f}" for name, p in row["probabilities"].items())


def format_console(result: EvaluationResult) -> str:
    """Human-readable summary for the terminal."""
    n = len(result.predictions)
    lines = [f"Accuracy: {result.n_correct} / {n} correct ({result.accuracy:.1%})", "",
             f"{'class':<14}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}"]
    for name in result.class_names:
        m = result.per_class[name]
        lines.append(f"{name:<14}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1-score']:>8.3f}{int(m['support']):>9}")
    width = max(len(c) for c in result.class_names) + 2
    lines += ["", "Confusion matrix (rows = true, columns = predicted):",
              " " * width + "".join(f"{c:>{width}}" for c in result.class_names)]
    for name, row in zip(result.class_names, result.confusion):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>{width}}" for v in row))
    wrong = result.misclassified()
    lines += ["", f"Misclassified ({len(wrong)}):"]
    lines += [f"  {_rel(r['path'])}: true={r['true']} predicted={r['predicted']} [{_probs(r)}]" for r in wrong] or ["  none"]
    if result.skipped:
        lines += ["", f"Skipped unreadable files ({len(result.skipped)}):"]
        lines += [f"  {_rel(p)}: {why}" for p, why in result.skipped]
    return "\n".join(lines)


def save_confusion_matrix(result: EvaluationResult, path: Path) -> None:
    """Render the confusion matrix as a PNG."""
    labels = [ALBUMS.get(c, c) for c in result.class_names]
    fig, ax = plt.subplots(figsize=(5, 4.4), dpi=120)
    ConfusionMatrixDisplay(result.confusion, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test set: {result.n_correct}/{len(result.predictions)} correct")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_thumbnail(source: Path, path: Path, size: int = 360) -> None:
    """Save an upright, downscaled copy of ``source`` for the report."""
    image = open_image(source)
    image.thumbnail((size, size))
    image.save(path, quality=90)


def write_report(result: EvaluationResult, report_dir: Path, metadata: Dict[str, Any]) -> Path:
    """Write ``evaluation.md``, ``evaluation.json``, ``confusion_matrix.png`` and ``difficult_example.jpg``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(result, report_dir / "confusion_matrix.png")
    hard = result.difficult_example()
    save_thumbnail(hard["path"], report_dir / "difficult_example.jpg")

    n = len(result.predictions)
    md = [
        "# Evaluation report", "",
        f"- Backbone: CLIP {metadata['backbone']['name']} (frozen), classifier: logistic regression, C = {metadata['classifier']['C']}",
        f"- Cross-validation on train ({metadata['training']['cv_folds']}-fold): "
        f"{metadata['training']['cv_accuracy_mean']:.1%} ± {metadata['training']['cv_accuracy_std']:.1%}",
        f"- **Test accuracy: {result.n_correct} / {n} correct ({result.accuracy:.1%})**", "",
        "## Per-class metrics", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for name in result.class_names:
        m = result.per_class[name]
        md.append(f"| {ALBUMS.get(name, name)} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {int(m['support'])} |")
    md += ["", "## Confusion matrix", "", "Rows are the true class, columns the predicted class.", "",
           "| true \\ predicted | " + " | ".join(ALBUMS.get(c, c) for c in result.class_names) + " |",
           "|---" * (len(result.class_names) + 1) + "|"]
    for name, row in zip(result.class_names, result.confusion):
        md.append(f"| {ALBUMS.get(name, name)} | " + " | ".join(str(v) for v in row) + " |")
    md += ["", "![Confusion matrix](confusion_matrix.png)", "", "## Misclassified images", ""]
    wrong = result.misclassified()
    if wrong:
        md += ["| File | True | Predicted | Probabilities |", "|---|---|---|---|"]
        md += [f"| `{_rel(r['path'])}` | {r['true']} | {r['predicted']} | {_probs(r)} |" for r in wrong]
    else:
        md.append("None.")
    md += ["", "## Lowest probability on the true class", "",
           "| File | True | Predicted | Probabilities |", "|---|---|---|---|"]
    md += [f"| `{_rel(r['path'])}` | {r['true']} | {r['predicted']} | {_probs(r)} |" for r in result.least_confident(5)]
    md += ["", "## Most difficult example", "", f"`{_rel(hard['path'])}`: true **{hard['true']}**, "
           f"predicted **{hard['predicted']}** ({_probs(hard)})", "", "![Difficult example](difficult_example.jpg)", ""]
    if result.skipped:
        md += ["## Skipped files", ""] + [f"- `{_rel(p)}`: {why}" for p, why in result.skipped] + [""]
    (report_dir / "evaluation.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "n_images": n, "n_correct": result.n_correct, "accuracy": result.accuracy,
        "per_class": result.per_class, "confusion_matrix": result.confusion.tolist(), "class_names": result.class_names,
        "misclassified": [{**r, "path": _rel(r["path"])} for r in wrong],
        "difficult_example": {**hard, "path": _rel(hard["path"])},
    }
    (report_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return report_dir / "evaluation.md"
