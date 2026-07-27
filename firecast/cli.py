from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


# ============================================================
# TABULAR TRAINING
# ============================================================

def train_tabular(args: argparse.Namespace) -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from firecast.data.tabular import feature_columns, load_tabular
    from firecast.evaluation import (
        binary_metrics,
        cross_validated_metrics,
        select_cost_threshold,
    )
    from firecast.artifacts import save_model_artifact
    from firecast.explainability import permutation_explanation
    from firecast.observability.experiment import write_experiment_manifest
    from firecast.models.tabular import model_factories
    from firecast.models.ensemble import weighted_probability_ensemble

    dataset = load_tabular(
        args.data,
        args.target,
        args.time_column,
    )

    columns = feature_columns(
        dataset,
        excluded=[args.time_column] if args.time_column else [],
    )

    x = dataset.frame[columns]
    y = dataset.frame[dataset.target].astype(int)

    # --------------------------------------------------------
    # Train / holdout split
    # --------------------------------------------------------

    if dataset.time_column:
        split_index = int(len(x) * 0.8)

        x_train = x.iloc[:split_index]
        x_holdout = x.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_holdout = y.iloc[split_index:]

        if y_train.nunique() < 2 or y_holdout.nunique() < 2:
            raise ValueError(
                "Chronological split must contain both classes "
                "in train and holdout"
            )

    else:
        (
            x_train,
            x_holdout,
            y_train,
            y_holdout,
        ) = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=args.seed,
        )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    categorical = x_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric = [
        column
        for column in columns
        if column not in categorical
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}
    fitted_pipelines = {}

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for name, estimator in model_factories(
        args.seed
    ).items():

        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        cv_metrics = cross_validated_metrics(
            pipeline,
            x_train,
            y_train,
            args.folds,
            args.seed,
        )

        pipeline.fit(
            x_train,
            y_train,
        )

        fitted_pipelines[name] = pipeline

        probabilities = pipeline.predict_proba(
            x_holdout
        )[:, 1]

        threshold = select_cost_threshold(
            y_holdout,
            probabilities,
            args.false_negative_cost,
            args.false_positive_cost,
        )

        results[name] = {
            "cross_validation": cv_metrics,
            "holdout": binary_metrics(
                y_holdout,
                probabilities,
                threshold,
            ),
        }

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_model = max(
        results,
        key=lambda name:
            results[name]["holdout"]["pr_auc"],
    )

    best_threshold = results[
        best_model
    ]["holdout"]["threshold"]

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble_probabilities = (
        weighted_probability_ensemble(
            fitted_pipelines,
            x_holdout,
        )
    )

    ensemble_threshold = select_cost_threshold(
        y_holdout,
        ensemble_probabilities,
        args.false_negative_cost,
        args.false_positive_cost,
    )

    results[
        "soft_voting_ensemble"
    ] = binary_metrics(
        y_holdout,
        ensemble_probabilities,
        ensemble_threshold,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": str(args.data),
        "target": args.target,
        "time_column": args.time_column,
        "seed": args.seed,
        "folds": args.folds,
        "python": sys.version,
        "platform": platform.platform(),
        "best_model_by_holdout_pr_auc": best_model,
        "feature_columns": columns,
        "threshold": best_threshold,
        "false_negative_cost": args.false_negative_cost,
        "false_positive_cost": args.false_positive_cost,
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    (
        output / "metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_model_artifact(
        output,
        fitted_pipelines[best_model],
        metadata,
    )

    write_experiment_manifest(
        output,
        args.data,
        metadata,
        args.seed,
        results,
    )


# ============================================================
# EXPLAIN TABULAR
# ============================================================

def explain_tabular(
    args: argparse.Namespace,
) -> None:

    from firecast.artifacts import (
        load_model_artifact,
    )

    from firecast.data.tabular import (
        load_tabular,
    )

    from firecast.explainability import (
        permutation_explanation,
    )

    pipeline, metadata = load_model_artifact(
        args.artifact
    )

    dataset = load_tabular(
        args.data,
        args.target,
    )

    columns = metadata[
        "feature_columns"
    ]

    explanation = permutation_explanation(
        pipeline,
        dataset.frame[columns],
        dataset.frame[
            args.target
        ].astype(int),
        repeats=args.repeats,
        seed=args.seed,
    )

    explanation.to_csv(
        Path(args.output),
        index=False,
    )


# ============================================================
# VALIDATE TABULAR
# ============================================================

def validate_tabular(
    args: argparse.Namespace,
) -> None:

    from firecast.data.tabular import (
        load_tabular,
    )

    from firecast.data.quality import (
        quality_report,
    )

    dataset = load_tabular(
        args.data,
        args.target,
        args.time_column,
        deduplicate=False,
    )

    report = quality_report(
        dataset.frame,
        args.target,
    )

    Path(
        args.output
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# FIRE SPREAD TRAINING
# ============================================================

def train_spread_command(
    args: argparse.Namespace,
) -> None:

    from firecast.spread.train import (
        train_spread,
    )

    metrics = train_spread(
        args.data,
        args.features.split(","),
        args.target,
        args.output,
        args.model,
        args.lookback,
        args.horizon,
        args.epochs,
        args.batch_size,
        args.seed,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


# ============================================================
# ONNX BENCHMARK
# ============================================================

def benchmark_onnx_command(
    args: argparse.Namespace,
) -> None:

    from firecast.deployment.benchmark import (
        benchmark_onnx,
    )

    result = benchmark_onnx(
        args.model,
        (
            args.batch,
            args.lookback,
            args.features,
        ),
        args.iterations,
        args.warmup,
    )

    Path(
        args.output
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


# ============================================================
# YOLO DETECTOR TRAINING
# ============================================================

def train_detector_command(
    args: argparse.Namespace,
) -> None:

    # IMPORTANT:
    # Import YOLO training code only when this command
    # is actually executed.
    #
    # This prevents pandas and other FireCast modules
    # from being imported when running train-detector.

    from firecast.vision.train import (
        train_detector,
    )

    train_detector(
        data_yaml=args.data,
        checkpoint=args.checkpoint,
        output=args.output,
        epochs=args.epochs,
        image_size=args.image_size,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )


# ============================================================
# YOLO DETECTOR EVALUATION
# ============================================================

def evaluate_detector_command(
    args: argparse.Namespace,
) -> None:

    from firecast.vision.evaluate import (
        evaluate_detector,
    )

    evaluate_detector(
        args.data,
        args.checkpoint,
        args.output,
        args.image_size,
        args.batch,
        args.device,
    )


# ============================================================
# YOLO DETECTOR COMPARISON
# ============================================================

def compare_detectors_command(
    args: argparse.Namespace,
) -> None:

    from firecast.vision.compare import (
        compare_checkpoints,
    )

    checkpoints = {}

    for item in args.checkpoint:

        name, path = item.split(
            "=",
            1,
        )

        checkpoints[name] = path

    compare_checkpoints(
        args.data,
        checkpoints,
        args.output,
        args.image_size,
        args.batch,
        args.device,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def health_command(
    args: argparse.Namespace,
) -> None:

    from firecast.observability.health import (
        readiness_report,
    )

    report = readiness_report(
        args.artifact,
        args.module,
    )

    Path(
        args.output
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


# ============================================================
# DRIFT DETECTION
# ============================================================

def drift_command(
    args: argparse.Namespace,
) -> None:

    import pandas as pd

    from firecast.observability.drift import (
        drift_report,
    )

    def load_dataset(path):

        # Skip extra title row in Algerian dataset
        df = pd.read_csv(
            path,
            skiprows=1,
        )

        # Clean column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(
                " ",
                "_",
                regex=False,
            )
        )

        return df

    reference = load_dataset(
        args.reference
    )

    current = load_dataset(
        args.current
    )

    columns = [
        c.strip()
        for c in args.columns.split(",")
    ]

    report = drift_report(
        reference,
        current,
        columns,
        args.threshold,
    )

    report.to_csv(
        args.output,
        index=False,
    )

    print(
        f"Drift report saved to: "
        f"{args.output}"
    )


# ============================================================
# TABULAR PREDICTION
# ============================================================

def predict_tabular_command(
    args: argparse.Namespace,
) -> None:

    from firecast.predict import (
        predict_file,
    )

    predictions = predict_file(
        args.artifact,
        args.input,
    )

    predictions.to_csv(
        args.output,
        index=False,
    )


# ============================================================
# MAIN CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        prog="firecast"
    )

    subparsers = (
        parser.add_subparsers(
            required=True
        )
    )

    # ========================================================
    # TRAIN TABULAR
    # ========================================================

    train = subparsers.add_parser(
        "train-tabular"
    )

    train.add_argument(
        "--data",
        required=True,
    )

    train.add_argument(
        "--target",
        required=True,
    )

    train.add_argument(
        "--output",
        required=True,
    )

    train.add_argument(
        "--time-column"
    )

    train.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    train.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    train.add_argument(
        "--false-negative-cost",
        type=float,
        default=5.0,
    )

    train.add_argument(
        "--false-positive-cost",
        type=float,
        default=1.0,
    )

    train.set_defaults(
        func=train_tabular
    )

    # ========================================================
    # EXPLAIN TABULAR
    # ========================================================

    explain = subparsers.add_parser(
        "explain-tabular"
    )

    explain.add_argument(
        "--artifact",
        required=True,
    )

    explain.add_argument(
        "--data",
        required=True,
    )

    explain.add_argument(
        "--target",
        required=True,
    )

    explain.add_argument(
        "--output",
        required=True,
    )

    explain.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    explain.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    explain.set_defaults(
        func=explain_tabular
    )

    # ========================================================
    # VALIDATE TABULAR
    # ========================================================

    validate = subparsers.add_parser(
        "validate-tabular"
    )

    validate.add_argument(
        "--data",
        required=True,
    )

    validate.add_argument(
        "--target",
        required=True,
    )

    validate.add_argument(
        "--output",
        required=True,
    )

    validate.add_argument(
        "--time-column"
    )

    validate.set_defaults(
        func=validate_tabular
    )

    # ========================================================
    # TRAIN SPREAD
    # ========================================================

    spread = subparsers.add_parser(
        "train-spread"
    )

    spread.add_argument(
        "--data",
        required=True,
    )

    spread.add_argument(
        "--features",
        required=True,
        help=(
            "Comma-separated feature columns"
        ),
    )

    spread.add_argument(
        "--target",
        required=True,
    )

    spread.add_argument(
        "--output",
        required=True,
    )

    spread.add_argument(
        "--model",
        choices=[
            "lstm",
            "gru",
            "transformer",
        ],
        default="transformer",
    )

    spread.add_argument(
        "--lookback",
        type=int,
        default=24,
    )

    spread.add_argument(
        "--horizon",
        type=int,
        default=6,
    )

    spread.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    spread.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    spread.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    spread.set_defaults(
        func=train_spread_command
    )

    # ========================================================
    # BENCHMARK ONNX
    # ========================================================

    benchmark = subparsers.add_parser(
        "benchmark-onnx"
    )

    benchmark.add_argument(
        "--model",
        required=True,
    )

    benchmark.add_argument(
        "--output",
        required=True,
    )

    benchmark.add_argument(
        "--batch",
        type=int,
        default=1,
    )

    benchmark.add_argument(
        "--lookback",
        type=int,
        required=True,
    )

    benchmark.add_argument(
        "--features",
        type=int,
        required=True,
    )

    benchmark.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    benchmark.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    benchmark.set_defaults(
        func=benchmark_onnx_command
    )

    # ========================================================
    # TRAIN YOLO DETECTOR
    # ========================================================

    detector_train = subparsers.add_parser(
        "train-detector"
    )

    detector_train.add_argument(
        "--data",
        required=True,
    )

    detector_train.add_argument(
        "--checkpoint",
        required=True,
    )

    detector_train.add_argument(
        "--output",
        required=True,
    )

    detector_train.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    detector_train.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    detector_train.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    detector_train.add_argument(
        "--device"
    )

    detector_train.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # NEW:
    # Resume training from last.pt
    detector_train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume YOLO training from "
            "the supplied checkpoint"
        ),
    )

    detector_train.set_defaults(
        func=train_detector_command
    )

    # ========================================================
    # EVALUATE YOLO DETECTOR
    # ========================================================

    detector_eval = subparsers.add_parser(
        "evaluate-detector"
    )

    detector_eval.add_argument(
        "--data",
        required=True,
    )

    detector_eval.add_argument(
        "--checkpoint",
        required=True,
    )

    detector_eval.add_argument(
        "--output",
        required=True,
    )

    detector_eval.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    detector_eval.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    detector_eval.add_argument(
        "--device"
    )

    detector_eval.set_defaults(
        func=evaluate_detector_command
    )

    # ========================================================
    # COMPARE YOLO DETECTORS
    # ========================================================

    detector_compare = subparsers.add_parser(
        "compare-detectors"
    )

    detector_compare.add_argument(
        "--data",
        required=True,
    )

    detector_compare.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "name=checkpoint_path"
        ),
    )

    detector_compare.add_argument(
        "--output",
        required=True,
    )

    detector_compare.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    detector_compare.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    detector_compare.add_argument(
        "--device"
    )

    detector_compare.set_defaults(
        func=compare_detectors_command
    )

    # ========================================================
    # HEALTH
    # ========================================================

    health = subparsers.add_parser(
        "health"
    )

    health.add_argument(
        "--artifact",
        required=True,
    )

    health.add_argument(
        "--output",
        required=True,
    )

    health.add_argument(
        "--module",
        action="append",
        default=[],
    )

    health.set_defaults(
        func=health_command
    )

    # ========================================================
    # DRIFT
    # ========================================================

    drift = subparsers.add_parser(
        "drift"
    )

    drift.add_argument(
        "--reference",
        required=True,
    )

    drift.add_argument(
        "--current",
        required=True,
    )

    drift.add_argument(
        "--columns",
        required=True,
        help=(
            "Comma-separated numeric "
            "feature columns"
        ),
    )

    drift.add_argument(
        "--threshold",
        type=float,
        default=0.2,
    )

    drift.add_argument(
        "--output",
        required=True,
    )

    drift.set_defaults(
        func=drift_command
    )

    # ========================================================
    # PREDICT TABULAR
    # ========================================================

    predict = subparsers.add_parser(
        "predict-tabular"
    )

    predict.add_argument(
        "--artifact",
        required=True,
    )

    predict.add_argument(
        "--input",
        required=True,
    )

    predict.add_argument(
        "--output",
        required=True,
    )

    predict.set_defaults(
        func=predict_tabular_command
    )

    # ========================================================
    # PARSE ARGUMENTS
    # ========================================================

    args = parser.parse_args()

    args.func(args)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()