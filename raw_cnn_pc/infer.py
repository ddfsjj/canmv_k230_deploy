import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Raw+CNN inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="infer_config.json",
        help="Path to inference config json (relative to script dir by default).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional override for test data directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Prediction csv output path (relative to script dir by default).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap of samples to run (e.g. 10 for quick smoke test).",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def resolve_positive_step(value, fallback: int, field_name: str) -> int:
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_label_from_name(filename: str) -> float:
    stem = Path(filename).stem
    if "-" not in stem:
        return np.nan
    token = stem.split("-")[0]
    try:
        return float(token)
    except ValueError:
        return np.nan


def read_signal(csv_path: Path):
    df = pd.read_csv(csv_path, header=None, usecols=[0])
    signal = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float32)
    return signal


def ensure_per_layer(value, num_layers: int, field: str):
    if isinstance(value, (list, tuple)):
        if len(value) != num_layers:
            raise ValueError(f"{field} length mismatch: {len(value)} vs {num_layers}")
        return list(value)
    return [value] * num_layers


class CNNAll(nn.Module):
    def __init__(self, input_shape, conv_filters, kernel_size=3, pool_size=2):
        super().__init__()
        time_steps, features = input_shape
        self.time_steps = int(time_steps)
        self.features = int(features)

        conv_filters = list(conv_filters)
        if not conv_filters:
            raise ValueError("conv_filters must not be empty.")
        num_layers = len(conv_filters)
        kernel_sizes = ensure_per_layer(kernel_size, num_layers, "kernel_size")
        pool_sizes = ensure_per_layer(pool_size, num_layers, "pool_size")
        in_channels = self.time_steps
        self.convs = nn.ModuleList()
        for out_channels, k, p in zip(conv_filters, kernel_sizes, pool_sizes):
            k = int(k)
            self.convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            in_channels = int(out_channels)
        self.pools = nn.ModuleList([nn.MaxPool1d(int(p)) for p in pool_sizes])

        length_after = self.features
        for p in pool_sizes:
            length_after = length_after // int(p)
        flatten_dim = int(conv_filters[-1]) * int(length_after)
        self.fc = nn.Linear(flatten_dim, 1)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
):
    X_list = []
    y_list = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = read_signal(csv_file)
        if signal.size < base_window_size:
            continue

        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            features.append(window.astype(np.float32))

        if len(features) < seq_length:
            continue

        label = parse_label_from_name(csv_file.name)
        for i in range(0, len(features) - seq_length + 1, seq_step):
            X_list.append(np.stack(features[i : i + seq_length], axis=0))
            y_list.append(label)

    if not X_list:
        return np.empty((0, seq_length, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def save_predictions(output_csv: Path, y_true: np.ndarray, y_pred: np.ndarray):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "prediction"])
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            writer.writerow([idx, float(t), float(p)])


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_json(config_path)

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["test_data_dir"])
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    model_path = Path(cfg["model"]["weights_path"])
    scaler_path = Path(cfg["normalization"]["scaler_path"])
    if not model_path.is_absolute():
        model_path = root / model_path
    if not scaler_path.is_absolute():
        scaler_path = root / scaler_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    base_window_size = require_positive_int(cfg["data"]["base_window_size"], "data.base_window_size")
    base_step_cfg = cfg["data"].get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window_size // 2, "data.base_step")
    seq_length = require_positive_int(cfg["data"]["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(cfg["data"]["sequence_step"], "data.sequence_step")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"max_samples must be > 0, got {args.max_samples}")

    t_total_start = time.perf_counter()
    X, y = build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"No valid samples found under: {data_dir}")
    if args.max_samples is not None:
        limit = min(int(args.max_samples), int(X.shape[0]))
        X = X[:limit]
        y = y[:limit]

    scaler = joblib.load(scaler_path)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape).astype(np.float32)

    device = torch.device("cpu")
    model = CNNAll(
        input_shape=tuple(X_scaled.shape[1:]),
        conv_filters=cfg["model"]["conv_filters"],
        kernel_size=cfg["model"]["kernel_size"],
        pool_size=cfg["model"]["pool_size"],
    )
    state_dict = load_state_dict_compat(model_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X_scaled).to(device)
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().reshape(-1)
    t_infer_end = time.perf_counter()
    t_total_end = time.perf_counter()

    save_predictions(output_path, y, y_pred)

    valid_mask = np.isfinite(y)
    mae = np.mean(np.abs(y_pred[valid_mask] - y[valid_mask])) if valid_mask.any() else float("nan")
    rmse = (
        float(np.sqrt(np.mean((y_pred[valid_mask] - y[valid_mask]) ** 2)))
        if valid_mask.any()
        else float("nan")
    )

    print("=== Raw + CNN Inference ===")
    print(f"data_dir: {data_dir}")
    print(f"model_path: {model_path}")
    print(f"scaler_path: {scaler_path}")
    print(f"samples: {X.shape[0]}")
    print(f"input_shape: {tuple(X.shape[1:])}")
    print(f"inference_time_sec: {t_infer_end - t_infer_start:.6f}")
    print(f"inference_time_per_sample_ms: {(t_infer_end - t_infer_start) * 1000 / X.shape[0]:.6f}")
    print(f"pipeline_total_time_sec: {t_total_end - t_total_start:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"prediction_csv: {output_path}")
    print("first_10_predictions:", np.round(y_pred[:10], 6).tolist())


if __name__ == "__main__":
    main()
