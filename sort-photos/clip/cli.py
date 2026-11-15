"""Command-line interface: ``python -m clip {download-model,train,evaluate,check-duplicates,sort}``."""
import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from . import config


def _int_at_least(minimum: int) -> Callable[[str], int]:
    """argparse type: an integer >= ``minimum``."""
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}, got {value}")
        return value
    return parse


def _probability(text: str) -> float:
    """argparse type: a float in [0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _offline() -> None:
    """Forbid hub libraries from reaching the network; our own loader only reads local files anyway."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def cmd_download_model(args: argparse.Namespace) -> int:
    from .clip import download

    path = download(args.backbone, args.backbone_dir)
    print(f"Backbone {args.backbone} cached at {path} (SHA256 verified). Other commands now work offline.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    groups_file = None if args.no_groups else (args.groups or config.default_groups_file(args.data))
    if groups_file is not None and not groups_file.is_file():
        raise FileNotFoundError(f"Groups file not found: {groups_file}")
    report = train(args.data, args.out, args.backbone, args.backbone_dir, args.device or config.default_device(),
                   args.batch_size, args.seed, args.folds, groups_file)
    print(f"\nTrained on {sum(report.train_counts.values())} images: {report.train_counts}")
    kind = f"grouped by venue/source ({report.n_groups} groups)" if report.cv_grouped else "stratified (no groups)"
    print(f"{report.cv_folds}-fold CV, {kind}, {report.cv_scoring} per setting:")
    for row in report.cv_results:
        chosen = row["C"] == report.best_c and row["class_weight"] == report.best_class_weight
        print(f"  C={row['C']:<7g} class_weight={str(row['class_weight']):<9} "
              f"{row['mean_score']:.3f} ± {row['std_score']:.3f}{'  <- selected' if chosen else ''}")
    if report.skipped:
        print(f"Skipped {len(report.skipped)} unreadable training files.")
    print(f"Saved {args.out / config.CLASSIFIER_FILE} and {args.out / config.METADATA_FILE}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, format_console, write_report
    from .predict import PhotoClassifier

    classifier = PhotoClassifier.load(args.model_dir, args.backbone_dir, args.device)
    result = evaluate(classifier, args.data, args.batch_size)
    print("\n" + format_console(result))
    report = write_report(result, args.report_dir, classifier.metadata)
    print(f"\nReport written to {report}")
    return 0


def _confidence(value: float) -> str:
    """3 decimals, truncated (not rounded) so a value just below the threshold never prints as equal to it."""
    return f"{math.floor(value * 1000) / 1000:.3f}"


def cmd_sort(args: argparse.Namespace) -> int:
    from .predict import PhotoClassifier
    from .sorter import NoImagesFound, sort_folder

    classifier = PhotoClassifier.load(args.model_dir, args.backbone_dir, args.device)
    try:
        result = sort_folder(args.input, args.output, classifier, args.recursive, args.min_confidence, args.batch_size,
                             append=args.append, unsure=args.unsure)
    except NoImagesFound as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"\nCopied {len(result.sorted_files)} photos into {result.output_dir} (originals unchanged):")
    for album, count in result.counts().items():
        print(f"  {album:<15}{count:>5}")
    if args.min_confidence > 0:
        print(f"Low confidence (< {args.min_confidence:g}): {len(result.low_confidence)}")
        for item in result.low_confidence:
            best = "" if item.album == item.predicted_album else f", best guess {item.predicted_album}"
            print(f"  {item.original_path} -> {item.album} ({_confidence(item.confidence)}{best})")
    print(f"Skipped: {len(result.skipped)}")
    for path, why in result.skipped:
        print(f"  {path}: {why}")
    added = f" ({result.previous_rows} earlier + {len(result.sorted_files)} new rows)" if args.append else ""
    print(f"Predictions: {result.output_dir / config.PREDICTIONS_FILE}{added}")
    if result.copy_failures:
        print(f"Copy failures: {len(result.copy_failures)} of {len(result.copy_failures) + len(result.sorted_files)} "
              "attempted copies failed (no incomplete files were left in the albums).", file=sys.stderr)
    if result.all_copies_failed:
        print("Error: every copy failed; nothing was sorted.", file=sys.stderr)
        return 1
    return 0


def cmd_check_duplicates(args: argparse.Namespace) -> int:
    from .duplicates import find_duplicates
    from .features import ClipEmbedder
    from .predict import load_artifact

    _, metadata = load_artifact(args.model_dir)
    embedder = ClipEmbedder(metadata["backbone"]["name"], args.backbone_dir, args.device)
    pairs, summary = find_duplicates(args.data, args.against, embedder, args.batch_size, args.cosine_threshold,
                                     args.dhash_distance)
    print(f"\nCompared {summary['candidates']} images in {args.data} with {summary['references']} reference images.")
    a, b = summary["max_cosine_pair"]
    print(f"Closest pair: cosine {summary['max_cosine']:.3f} ({a} ~ {b}); smallest dHash distance: {summary['min_dhash_distance']}")
    print(f"Duplicates / near-duplicates (sha256 equal, dHash <= {args.dhash_distance}, or cosine >= {args.cosine_threshold}): {len(pairs)}")
    for pair in pairs:
        print(f"  {pair.candidate} ~ {pair.reference}: sha256_equal={pair.sha256_equal} "
              f"dhash={pair.dhash_distance} cosine={pair.cosine:.3f}")
    return 1 if pairs else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m clip", description="Sort café photos into Food & Drinks, People and Interior albums.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, model: bool = True) -> None:
        p.add_argument("--backbone-dir", type=Path, default=config.DEFAULT_BACKBONE_DIR, help="local folder with the backbone weights")
        p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
        p.add_argument("--batch-size", type=_int_at_least(1), default=config.BATCH_SIZE, help="images per batch (positive integer)")
        if model:
            p.add_argument("--model-dir", type=Path, default=config.DEFAULT_MODEL_DIR, help="folder with the trained classifier")

    p = sub.add_parser("download-model", help="one-time download of the backbone weights")
    p.add_argument("--backbone", default=config.BACKBONE_NAME)
    p.add_argument("--backbone-dir", type=Path, default=config.DEFAULT_BACKBONE_DIR)
    p.set_defaults(func=cmd_download_model)

    p = sub.add_parser("train", help="train the classifier on labelled photos")
    p.add_argument("--data", type=Path, default=config.DEFAULT_TRAIN_DIR, help="folder with one sub-folder per class")
    p.add_argument("--out", type=Path, default=config.DEFAULT_MODEL_DIR)
    p.add_argument("--backbone", default=config.BACKBONE_NAME)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--folds", type=_int_at_least(2), default=config.CV_FOLDS, help="CV folds (>= 2)")
    p.add_argument("--groups", type=Path, default=None,
                   help="CSV with class,filename,group columns for venue/source-grouped CV "
                        "(default: <data>/../manifest.csv if it has a 'group' column)")
    p.add_argument("--no-groups", action="store_true", help="use plain stratified CV")
    common(p, model=False)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate on held-out labelled photos")
    p.add_argument("--data", type=Path, default=config.DEFAULT_TEST_DIR)
    p.add_argument("--report-dir", type=Path, default=config.DEFAULT_REPORT_DIR)
    common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("check-duplicates", help="find duplicates/near-duplicates of an evaluation set in other folders")
    p.add_argument("--data", type=Path, required=True, help="evaluation images (searched recursively)")
    p.add_argument("--against", type=Path, nargs="+", default=[config.DEFAULT_TRAIN_DIR, config.DEFAULT_TEST_DIR])
    p.add_argument("--cosine-threshold", type=_probability, default=0.95)
    p.add_argument("--dhash-distance", type=_int_at_least(0), default=6)
    common(p)
    p.set_defaults(func=cmd_check_duplicates)

    p = sub.add_parser("sort", help="copy photos from a folder into album folders")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recursive", action="store_true", help="include photos in sub-folders")
    p.add_argument("--min-confidence", type=_probability, default=0.0, help="flag predictions below this probability")
    p.add_argument("--unsure", action="store_true",
                   help=f"copy photos below --min-confidence to an '{config.UNSURE_ALBUM}' album instead of their best album")
    p.add_argument("--append", action="store_true",
                   help="add to an output folder from a previous run (keeps its files, extends predictions.csv)")
    common(p)
    p.set_defaults(func=cmd_sort)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if args.command != "download-model":
        _offline()
    config.set_seed(getattr(args, "seed", config.SEED))
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
