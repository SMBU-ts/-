import argparse
import csv
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - handled at runtime
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as exc:  # pragma: no cover - handled at runtime
    mean_absolute_error = None
    mean_squared_error = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order
except ImportError as exc:  # pragma: no cover - handled at runtime
    VAR = None
    VECM = None
    coint_johansen = None
    select_order = None
    STATSMODELS_IMPORT_ERROR = exc
else:
    STATSMODELS_IMPORT_ERROR = None

TARGETS = ("rtt", "cwnd")
MODELS = ("VAR", "VECM")
WINDOWS = ("expanding", "sliding")


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    train_size: int = 2880
    test_size: int = 720
    sliding_size: int = 720
    maxlags: int = 10
    seed: int = 42
    max_files: Optional[int] = None
    test_steps: Optional[int] = None
    skip_existing: bool = False


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_bbr_forecasting.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def require_dependencies() -> None:
    missing = []
    if MATPLOTLIB_IMPORT_ERROR is not None:
        missing.append(f"matplotlib ({MATPLOTLIB_IMPORT_ERROR})")
    if SKLEARN_IMPORT_ERROR is not None:
        missing.append(f"scikit-learn ({SKLEARN_IMPORT_ERROR})")
    if STATSMODELS_IMPORT_ERROR is not None:
        missing.append(f"statsmodels ({STATSMODELS_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Missing required dependencies:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\nInstall them with: python -m pip install statsmodels scikit-learn matplotlib"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def natural_sort_key(path: Path) -> Tuple[str, int]:
    stem = path.name.split(".")[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    prefix = "".join(ch for ch in stem if not ch.isdigit())
    return prefix, int(digits or 0)


def load_bbr_files(config: ExperimentConfig) -> List[Tuple[str, np.ndarray]]:
    files = sorted(config.data_dir.glob("*.csv"), key=natural_sort_key)
    if config.max_files is not None:
        files = files[: config.max_files]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {config.data_dir}")

    series = []
    for file_path in files:
        df = pd.read_csv(file_path, sep=";")
        df = df.drop(columns=["Unnamed: 19"], errors="ignore")
        missing_targets = [target for target in TARGETS if target not in df.columns]
        if missing_targets:
            raise ValueError(f"{file_path} is missing target columns: {missing_targets}")
        if len(df) != config.train_size + config.test_size:
            raise ValueError(
                f"{file_path} has {len(df)} rows; expected "
                f"{config.train_size + config.test_size}"
            )
        values = df.loc[:, TARGETS].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{file_path} contains non-finite values in {TARGETS}")
        series.append((file_path.name, values))

    logging.info("Loaded %d bbr CSV files from %s", len(series), config.data_dir)
    return series


def select_var_lag(history: np.ndarray, maxlags: int) -> int:
    usable_maxlags = max(1, min(maxlags, len(history) // 5, len(history) - 2))
    try:
        selected = VAR(history).select_order(usable_maxlags).aic
        if selected is None or selected <= 0:
            return 1
        return int(selected)
    except Exception:
        return 1


def forecast_var_one_step(history: np.ndarray, maxlags: int) -> np.ndarray:
    lag_order = select_var_lag(history, maxlags)
    try:
        fit = VAR(history).fit(lag_order)
        return np.asarray(fit.forecast(history[-lag_order:], steps=1)[0], dtype=float)
    except Exception as exc:
        logging.debug("VAR failed with lag %s: %s; using last observation", lag_order, exc)
        return history[-1].copy()


def select_vecm_params(history: np.ndarray, maxlags: int) -> Tuple[int, int]:
    usable_maxlags = max(1, min(maxlags, len(history) // 5, len(history) - 3))
    try:
        order = select_order(history, maxlags=usable_maxlags, deterministic="ci")
        k_ar_diff = int(order.aic) if order.aic is not None and order.aic >= 1 else 1
    except Exception:
        k_ar_diff = 1

    try:
        johansen = coint_johansen(history, det_order=0, k_ar_diff=k_ar_diff)
        rank = int(np.sum(johansen.lr1 > johansen.cvt[:, 1]))
        coint_rank = min(max(rank, 1), history.shape[1] - 1)
    except Exception:
        coint_rank = 1

    return k_ar_diff, coint_rank


def forecast_vecm_one_step(history: np.ndarray, maxlags: int) -> np.ndarray:
    k_ar_diff, coint_rank = select_vecm_params(history, maxlags)
    try:
        fit = VECM(
            history,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic="ci",
        ).fit()
        return np.asarray(fit.predict(steps=1)[0], dtype=float)
    except Exception as exc:
        logging.debug(
            "VECM failed with k_ar_diff=%s coint_rank=%s: %s; using last observation",
            k_ar_diff,
            coint_rank,
            exc,
        )
        return history[-1].copy()


def history_for_step(
    values: np.ndarray,
    step: int,
    config: ExperimentConfig,
    window: str,
) -> np.ndarray:
    current_end = config.train_size + step
    if window == "expanding":
        start = 0
    elif window == "sliding":
        start = max(0, current_end - config.sliding_size)
    else:
        raise ValueError(f"Unknown window mode: {window}")
    return values[start:current_end]


def forecast_model(
    model_name: str,
    history: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    if model_name == "VAR":
        return forecast_var_one_step(history, config.maxlags)
    if model_name == "VECM":
        return forecast_vecm_one_step(history, config.maxlags)
    raise ValueError(f"Unknown model: {model_name}")


def run_forecasts(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_dependencies()
    set_seed(config.seed)
    loaded = load_bbr_files(config)
    test_steps = config.test_steps or config.test_size
    if test_steps > config.test_size:
        raise ValueError(f"--test-steps cannot exceed {config.test_size}")

    predictions: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for file_index, (file_name, values) in enumerate(loaded, start=1):
        actual_test = values[config.train_size : config.train_size + test_steps]
        logging.info("Processing %s (%d/%d)", file_name, file_index, len(loaded))

        for window in WINDOWS:
            for model_name in MODELS:
                logging.info("  %s / %s", model_name, window)
                model_preds: List[np.ndarray] = []

                for step in range(test_steps):
                    history = history_for_step(values, step, config, window)
                    try:
                        pred = forecast_model(
                            model_name,
                            history,
                            config,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "file": file_name,
                                "step": step + 1,
                                "model": model_name,
                                "window": window,
                                "error": repr(exc),
                            }
                        )
                        logging.exception(
                            "Failed %s / %s / %s step %d",
                            file_name,
                            model_name,
                            window,
                            step + 1,
                        )
                        pred = np.array([np.nan, np.nan], dtype=float)
                    model_preds.append(pred)

                pred_array = np.asarray(model_preds, dtype=float)
                for step, (actual, pred) in enumerate(zip(actual_test, pred_array), start=1):
                    predictions.append(
                        {
                            "file": file_name,
                            "step": step,
                            "model": model_name,
                            "window": window,
                            "actual_rtt": actual[0],
                            "pred_rtt": pred[0],
                            "actual_cwnd": actual[1],
                            "pred_cwnd": pred[1],
                        }
                    )

    predictions_df = pd.DataFrame(predictions)
    failures_df = pd.DataFrame(failures)
    if not failures_df.empty:
        failures_path = config.output_dir / "failures_bbr.csv"
        failures_df.to_csv(failures_path, index=False, encoding="utf-8")
        logging.warning("Recorded %d failures in %s", len(failures_df), failures_path)
    else:
        logging.info("No model-step failures recorded")

    metrics_df = compute_metrics(predictions_df)
    return predictions_df, metrics_df


def compute_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        for window in WINDOWS:
            subset = predictions_df[
                (predictions_df["model"] == model_name)
                & (predictions_df["window"] == window)
            ]
            for target in TARGETS:
                actual_col = f"actual_{target}"
                pred_col = f"pred_{target}"
                clean = subset[[actual_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if clean.empty:
                    rmse = math.nan
                    mae = math.nan
                else:
                    rmse = math.sqrt(mean_squared_error(clean[actual_col], clean[pred_col]))
                    mae = mean_absolute_error(clean[actual_col], clean[pred_col])
                rows.append(
                    {
                        "model": model_name,
                        "window": window,
                        "target": target,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )
    return pd.DataFrame(rows)


def plot_average_predictions(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for window in WINDOWS:
        for target in TARGETS:
            plt.figure(figsize=(12, 6))
            actual_col = f"actual_{target}"
            pred_col = f"pred_{target}"

            actual = (
                predictions_df[predictions_df["window"] == window]
                .groupby("step")[actual_col]
                .mean()
                .sort_index()
            )
            plt.plot(actual.index, actual.values, label=f"Actual {target}", color="black", linewidth=2)

            for model_name in MODELS:
                pred = (
                    predictions_df[
                        (predictions_df["window"] == window)
                        & (predictions_df["model"] == model_name)
                    ]
                    .groupby("step")[pred_col]
                    .mean()
                    .sort_index()
                )
                plt.plot(pred.index, pred.values, label=model_name, linewidth=1.6)

            plt.title(f"Average bbr {target} forecast ({window} window)")
            plt.xlabel("Test step")
            plt.ylabel(target)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            output_path = output_dir / f"avg_{target}_{window}.png"
            plt.savefig(output_path, dpi=160)
            plt.close()
            logging.info("Wrote %s", output_path)


def validate_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    expected_files = config.max_files or 50
    expected_steps = config.test_steps or config.test_size
    expected_prediction_rows = expected_files * len(MODELS) * len(WINDOWS) * expected_steps
    if len(predictions_df) != expected_prediction_rows:
        raise AssertionError(
            f"Expected {expected_prediction_rows} prediction rows, got {len(predictions_df)}"
        )

    if len(metrics_df) != len(MODELS) * len(WINDOWS) * len(TARGETS):
        expected_metric_rows = len(MODELS) * len(WINDOWS) * len(TARGETS)
        raise AssertionError(
            f"Expected {expected_metric_rows} metric rows, got {len(metrics_df)}"
        )

    metric_values = metrics_df[["rmse", "mae"]].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise AssertionError("RMSE/MAE contains non-finite values")

    for window in WINDOWS:
        for target in TARGETS:
            image_path = config.output_dir / f"avg_{target}_{window}.png"
            if not image_path.exists() or image_path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty image: {image_path}")

    logging.info("Output validation passed")


def write_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "predictions_bbr.csv"
    metrics_path = config.output_dir / "metrics_bbr.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote %s", predictions_path)
    logging.info("Wrote %s", metrics_path)
    plot_average_predictions(predictions_df, config.output_dir)
    validate_outputs(predictions_df, metrics_df, config)


def write_runtime(output_dir: Path, elapsed_seconds: float) -> None:
    runtime_path = output_dir / "runtime_bbr.txt"
    runtime_path.write_text(
        "BBR forecasting program runtime\n"
        f"Seconds: {elapsed_seconds:.3f}\n"
        f"HH:MM:SS: {format_duration(elapsed_seconds)}\n",
        encoding="utf-8",
    )
    logging.info("Wrote %s", runtime_path)


def parse_args(argv: Optional[Iterable[str]] = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run VAR and VECM forecasts for D:\\ALL\\bbr data."
    )
    parser.add_argument("--data-dir", default=r"D:\ALL\bbr", type=Path)
    parser.add_argument("--output-dir", default=r"D:\ALL\results_bbr", type=Path)
    parser.add_argument("--train-size", default=2880, type=int)
    parser.add_argument("--test-size", default=720, type=int)
    parser.add_argument("--sliding-size", default=720, type=int)
    parser.add_argument("--maxlags", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-files", default=None, type=int)
    parser.add_argument("--test-steps", default=None, type=int)
    args = parser.parse_args(argv)

    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        sliding_size=args.sliding_size,
        maxlags=args.maxlags,
        seed=args.seed,
        max_files=args.max_files,
        test_steps=args.test_steps,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    start_time = time.perf_counter()
    config = parse_args(argv)
    configure_logging(config.output_dir)
    logging.info("Starting bbr forecasting experiment with config: %s", config)
    try:
        predictions_df, metrics_df = run_forecasts(config)
        write_outputs(predictions_df, metrics_df, config)
    except Exception:
        elapsed = time.perf_counter() - start_time
        logging.exception(
            "Experiment failed after %.3f seconds (%s)",
            elapsed,
            format_duration(elapsed),
        )
        return 1
    elapsed = time.perf_counter() - start_time
    write_runtime(config.output_dir, elapsed)
    logging.info(
        "Experiment completed successfully in %.3f seconds (%s)",
        elapsed,
        format_duration(elapsed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
