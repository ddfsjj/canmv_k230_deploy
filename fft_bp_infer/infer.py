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
    parser = argparse.ArgumentParser(description="Standalone FFT+BP inference.")
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


def activation_layer(name: str) -> nn.Module:
    key = (name or "relu").lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(0.01)
    if key in {"none", "linear", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class BPNet(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_units,
        activation="tanh",
        output_activation="none",
    ):
        super().__init__()
        time_steps, features = input_shape
        input_dim = int(time_steps) * int(features)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if not hidden_units:
            raise ValueError("hidden_units must not be empty.")

        if isinstance(activation, (list, tuple)):
            activations = list(activation)
        else:
            activations = [activation] * len(hidden_units)

        layers = []
        last_dim = input_dim
        for i, units in enumerate(hidden_units):
            layers.append(nn.Linear(last_dim, int(units)))
            act_name = activations[i] if i < len(activations) else activations[-1]
            layers.append(activation_layer(act_name))
            last_dim = int(units)

        layers.append(nn.Linear(last_dim, 1))
        out_act = activation_layer(output_activation)
        if not isinstance(out_act, nn.Identity):
            layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_bp_state_dict_keys(state_dict: dict):
    # Backward compatibility for checkpoints that include Flatten in nn.Sequential:
    # old keys: net.1.*, net.3.*  -> new keys: net.0.*, net.2.*
    if "net.1.weight" in state_dict and "net.0.weight" not in state_dict:
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("net.1."):
                mapped["net.0." + k.split("net.1.", 1)[1]] = v
            elif k.startswith("net.3."):
                mapped["net.2." + k.split("net.3.", 1)[1]] = v
            else:
                mapped[k] = v
        return mapped
    return state_dict


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


def resolve_nfft(nfft_cfg, window_size: int):
    if isinstance(nfft_cfg, int) and nfft_cfg > 0:
        return int(nfft_cfg)
    return 1 << (window_size - 1).bit_length()


def get_window(window_name: str, window_size: int):
    name = str(window_name).lower()
    if name in {"hann", "hanning"}:
        return np.hanning(window_size)
    if name == "hamming":
        return np.hamming(window_size)
    if name in {"rect", "boxcar", "none"}:
        return np.ones(window_size)
    raise ValueError(f"Unsupported FFT window: {window_name}")


def fft_transform(data_1d: np.ndarray, fft_cfg: dict):
    x = np.asarray(data_1d, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=np.float32)

    x = x - np.mean(x)
    x = x * get_window(fft_cfg.get("window", "hann"), x.size)
    nfft = resolve_nfft(fft_cfg.get("nfft", "auto"), x.size)
    y = np.fft.fft(x, n=nfft)
    p2 = np.abs(y / nfft)
    p1 = p2[: nfft // 2 + 1]
    if p1.size > 2:
        p1[1:-1] *= 2.0

    fs = float(fft_cfg.get("fs", 1000.0))
    f_min = float(fft_cfg.get("f_min", 0.1))
    f_max = float(fft_cfg.get("f_max", 50.0))
    freqs = fs * np.arange(0, nfft // 2 + 1) / nfft
    mask = (freqs >= f_min) & (freqs < f_max)
    return p1[mask].astype(np.float32)


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    fft_cfg: dict,
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
            feat = fft_transform(window, fft_cfg)
            if feat.size > 0:
                features.append(feat)

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
    fft_cfg = cfg["preprocessing"]["fft_config"]

    t_total_start = time.perf_counter()
    X, y = build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        fft_cfg=fft_cfg,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"No valid samples found under: {data_dir}")

    scaler = joblib.load(scaler_path)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape).astype(np.float32)

    device = torch.device("cpu")
    model = BPNet(
        input_shape=tuple(X_scaled.shape[1:]),
        hidden_units=cfg["model"]["hidden_units"],
        activation=cfg["model"]["activation"],
        output_activation=cfg["model"].get("output_activation", "none"),
    )
    state_dict = normalize_bp_state_dict_keys(load_state_dict_compat(model_path, device))
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

    print("=== FFT + BP Inference ===")
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
