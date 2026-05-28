import argparse
import csv
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as exc:  # pragma: no cover - handled at runtime
    mean_absolute_error = None
    mean_squared_error = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


TARGETS = ("rtt", "cwnd")
MODEL_NAME = "GRU"
WINDOWS = ("expanding", "sliding")


@dataclass
class GRUConfig:
    data_dir: Path
    output_dir: Path
    train_size: int = 3240
    test_size: int = 360
    forecast_horizon: int = 50
    lag: int = 15
    sliding_size: int = 360
    seed: int = 42
    hidden_size: int = 32
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 0.001
    max_files: Optional[int] = None
    test_steps: Optional[int] = None


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.output(gru_out[:, -1, :])


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_bbr_gru_forecasting.log"
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
    if TORCH_IMPORT_ERROR is not None:
        missing.append(f"torch ({TORCH_IMPORT_ERROR})")
    if SKLEARN_IMPORT_ERROR is not None:
        missing.append(f"scikit-learn ({SKLEARN_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Missing required dependencies:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\nUse the Anaconda launcher if available: py -3 D:\\ALL\\run_bbr_gru_forecasting.py"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def natural_sort_key(path: Path) -> Tuple[str, int]:
    stem = path.name.split(".")[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    prefix = "".join(ch for ch in stem if not ch.isdigit())
    return prefix, int(digits or 0)


def load_bbr_files(config: GRUConfig) -> List[Tuple[str, np.ndarray]]:
    files = sorted(config.data_dir.glob("*.csv"), key=natural_sort_key)
    if config.max_files is not None:
        files = files[: config.max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {config.data_dir}")

    expected_rows = config.train_size + config.test_size
    series = []
    for file_path in files:
        df = pd.read_csv(file_path, sep=";")
        df = df.drop(columns=["Unnamed: 19"], errors="ignore")
        missing_targets = [target for target in TARGETS if target not in df.columns]
        if missing_targets:
            raise ValueError(f"{file_path} is missing target columns: {missing_targets}")
        if len(df) != expected_rows:
            raise ValueError(f"{file_path} has {len(df)} rows; expected {expected_rows}")
        values = df.loc[:, TARGETS].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{file_path} contains non-finite values in {TARGETS}")
        series.append((file_path.name, values))

    logging.info("Loaded %d BBR CSV files from %s", len(series), config.data_dir)
    return series


def training_start_for_window(config: GRUConfig, window: str) -> int:
    if window == "expanding":
        return 0
    if window == "sliding":
        return config.train_size - config.sliding_size
    raise ValueError(f"Unknown window: {window}")


def make_supervised(values: np.ndarray, start: int, end: int, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for target_index in range(start + lag, end):
        x_rows.append(values[target_index - lag : target_index])
        y_rows.append(values[target_index])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def make_test(values: np.ndarray, config: GRUConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_steps = config.test_steps or config.test_size
    x_rows = []
    y_rows = []
    target_steps = []
    for target_index in range(config.train_size, config.train_size + test_steps):
        x_rows.append(values[target_index - config.lag : target_index])
        y_rows.append(values[target_index])
        target_steps.append(target_index - config.train_size + 1)
    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(target_steps, dtype=int),
    )


def standardize_train_test(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = train_x.reshape(-1, len(TARGETS)).mean(axis=0)
    x_std = train_x.reshape(-1, len(TARGETS)).std(axis=0)
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)

    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    train_x_scaled = (train_x - x_mean) / x_std
    train_y_scaled = (train_y - y_mean) / y_std
    test_x_scaled = (test_x - x_mean) / x_std
    return train_x_scaled, train_y_scaled, test_x_scaled, y_mean, y_std


def train_gru(train_x: np.ndarray, train_y: np.ndarray, config: GRUConfig) -> GRUForecaster:
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = GRUForecaster(hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)
        if epoch in {1, 10, 20, 40, config.epochs}:
            logging.info("    Epoch %d/%d loss=%.6f", epoch, config.epochs, epoch_loss / len(dataset))

    return model


def predict(model: GRUForecaster, test_x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(test_x, dtype=torch.float32)).numpy()
    return pred_scaled * y_std + y_mean


def compute_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        subset = predictions_df[predictions_df["window"] == window]
        for target in TARGETS:
            actual_col = f"actual_{target}"
            pred_col = f"pred_{target}"
            clean = subset[[actual_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
            rmse = math.sqrt(mean_squared_error(clean[actual_col], clean[pred_col]))
            mae = mean_absolute_error(clean[actual_col], clean[pred_col])
            rows.append(
                {
                    "model": MODEL_NAME,
                    "window": window,
                    "target": target,
                    "rmse": rmse,
                    "mae": mae,
                }
            )
    return pd.DataFrame(rows)


def build_predictions_df(
    file_name: str,
    window: str,
    target_steps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    config: GRUConfig,
) -> pd.DataFrame:
    rows = []
    for row_index, target_step in enumerate(target_steps):
        rows.append(
            {
                "file": file_name,
                "origin_step": int(target_step),
                "target_step": int(target_step),
                "horizon": config.forecast_horizon,
                "model": MODEL_NAME,
                "window": window,
                "actual_rtt": actual[row_index, 0],
                "pred_rtt": predicted[row_index, 0],
                "actual_cwnd": actual[row_index, 1],
                "pred_cwnd": predicted[row_index, 1],
            }
        )
    return pd.DataFrame(rows)


def run_forecasts(config: GRUConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_dependencies()
    # if config.forecast_horizon != 1:
    #     raise ValueError("This GRU script implements one-step forecasting only; use --forecast-horizon 1")
    if config.lag < 1:
        raise ValueError("--lag must be at least 1")
    if config.test_steps is not None and config.test_steps > config.test_size:
        raise ValueError("--test-steps cannot exceed --test-size")
    if config.sliding_size <= config.lag:
        raise ValueError("--sliding-size must be greater than --lag")
    if config.sliding_size > config.train_size:
        raise ValueError("--sliding-size cannot exceed --train-size")

    loaded = load_bbr_files(config)
    prediction_frames = []

    for file_index, (file_name, values) in enumerate(loaded, start=1):
        logging.info("Processing %s (%d/%d)", file_name, file_index, len(loaded))
        test_x, test_y, target_steps = make_test(values, config)

        for window in WINDOWS:
            train_start = training_start_for_window(config, window)
            train_x, train_y = make_supervised(values, train_start, config.train_size, config.lag)
            train_x_scaled, train_y_scaled, test_x_scaled, y_mean, y_std = standardize_train_test(
                train_x,
                train_y,
                test_x,
            )
            logging.info(
                "  %s / %s training_start=%d training_samples=%d test_samples=%d",
                MODEL_NAME,
                window,
                train_start,
                len(train_x_scaled),
                len(test_x_scaled),
            )
            set_seed(config.seed)
            model = train_gru(train_x_scaled, train_y_scaled, config)
            predicted = predict(model, test_x_scaled, y_mean, y_std)
            prediction_frames.append(
                build_predictions_df(file_name, window, target_steps, test_y, predicted, config)
            )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = compute_metrics(predictions_df)
    return predictions_df, metrics_df


def plot_average_predictions(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    for window in WINDOWS:
        window_subset = predictions_df[predictions_df["window"] == window]
        for target in TARGETS:
            actual_col = f"actual_{target}"
            pred_col = f"pred_{target}"
            actual = window_subset.groupby("target_step")[actual_col].mean().sort_index()
            pred = window_subset.groupby("target_step")[pred_col].mean().sort_index()

            plt.figure(figsize=(12, 6))
            plt.plot(actual.index, actual.values, label=f"Actual {target}", color="black", linewidth=2)
            plt.plot(pred.index, pred.values, label=f"{MODEL_NAME} forecast", linewidth=1.6)
            plt.title(f"Average BBR {target} GRU forecast ({window} window)")
            plt.xlabel("Target test step")
            plt.ylabel(target)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            output_path = output_dir / f"gru_avg_{target}_{window}.png"
            plt.savefig(output_path, dpi=160)
            plt.close()
            logging.info("Wrote %s", output_path)


def validate_outputs(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, config: GRUConfig) -> None:
    expected_files = config.max_files or len(predictions_df["file"].unique())
    expected_steps = config.test_steps or config.test_size
    expected_prediction_rows = expected_files * len(WINDOWS) * expected_steps
    expected_metric_rows = len(WINDOWS) * len(TARGETS)

    if len(predictions_df) != expected_prediction_rows:
        raise AssertionError(
            f"Expected {expected_prediction_rows} prediction rows, got {len(predictions_df)}"
        )
    if len(metrics_df) != expected_metric_rows:
        raise AssertionError(f"Expected {expected_metric_rows} metric rows, got {len(metrics_df)}")
    if not np.isfinite(metrics_df[["rmse", "mae"]].to_numpy(dtype=float)).all():
        raise AssertionError("Metrics contain non-finite values")
    for window in WINDOWS:
        for target in TARGETS:
            image_path = config.output_dir / f"avg_{target}_{window}.png"
            if not image_path.exists() or image_path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty image: {image_path}")
    logging.info("Output validation passed")


def write_outputs(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, config: GRUConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "predictions_bbr_gru.csv"
    metrics_path = config.output_dir / "metrics_bbr_gru.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    logging.info("Wrote %s", predictions_path)
    logging.info("Wrote %s", metrics_path)
    plot_average_predictions(predictions_df, config.output_dir)
    validate_outputs(predictions_df, metrics_df, config)


def write_runtime(output_dir: Path, elapsed_seconds: float) -> None:
    runtime_path = output_dir / "runtime_bbr_gru.txt"
    runtime_path.write_text(
        "BBR GRU forecasting program runtime\n"
        f"Seconds: {elapsed_seconds:.3f}\n"
        f"HH:MM:SS: {format_duration(elapsed_seconds)}\n",
        encoding="utf-8",
    )
    logging.info("Wrote %s", runtime_path)


def parse_args(argv: Optional[Iterable[str]] = None) -> GRUConfig:
    parser = argparse.ArgumentParser(description="Run GRU forecasts for D:\\ALL\\bbr data.")
    parser.add_argument("--data-dir", default=r"D:\ALL\bbr", type=Path)
    parser.add_argument("--output-dir", default=r"D:\ALL\results_bbr_gru", type=Path)
    parser.add_argument("--train-size", default=3240, type=int)
    parser.add_argument("--test-size", default=360, type=int)
    parser.add_argument("--forecast-horizon", default=50, type=int)
    parser.add_argument("--lag", default=15, type=int)
    parser.add_argument("--sliding-size", default=360, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--hidden-size", default=32, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--max-files", default=None, type=int)
    parser.add_argument("--test-steps", default=None, type=int)
    args = parser.parse_args(argv)

    return GRUConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        forecast_horizon=args.forecast_horizon,
        lag=args.lag,
        sliding_size=args.sliding_size,
        seed=args.seed,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_files=args.max_files,
        test_steps=args.test_steps,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    start_time = time.perf_counter()
    config = parse_args(argv)
    configure_logging(config.output_dir)
    logging.info("Starting BBR GRU forecasting experiment with config: %s", config)

    try:
        predictions_df, metrics_df = run_forecasts(config)
        write_outputs(predictions_df, metrics_df, config)
    except Exception:
        elapsed = time.perf_counter() - start_time
        logging.exception("BBR GRU experiment failed after %.3f seconds (%s)", elapsed, format_duration(elapsed))
        return 1

    elapsed = time.perf_counter() - start_time
    write_runtime(config.output_dir, elapsed)
    logging.info("BBR GRU experiment completed successfully in %.3f seconds (%s)", elapsed, format_duration(elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
