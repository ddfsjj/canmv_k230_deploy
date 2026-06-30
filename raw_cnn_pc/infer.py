import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from raw_cnn import config as common_config
from raw_cnn import data as common_data
from raw_cnn import models as common_models
from raw_cnn import scaler as common_scaler

"""
PC 端原始 CNN / CNN-LSTM 推理脚本。

主要用途：
1. 读取本地测试 CSV，按配置切窗和组序列。
2. 加载 `scaler.pkl` 做标准化。
3. 加载 `.pth` 权重执行推理。
4. 输出预测 CSV，并打印 MAE / RMSE 等指标。
"""
DEFAULT_CONFIG_PATH = "configs/infer/infer_config_cnn_tcn.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Raw+CNN inference.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
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
    return common_config.load_json(path)


def require_positive_int(value, field_name: str) -> int:
    return common_config.require_positive_int(value, field_name)


def resolve_positive_step(value, fallback: int, field_name: str) -> int:
    return common_config.resolve_positive_step(value, fallback, field_name)


def resolve_max_samples(cli_value, config_value):
    return common_config.resolve_max_samples(cli_value, config_value)


def load_state_dict_compat(path: Path, device: torch.device):
    return common_models.load_state_dict_compat(path, device)


def parse_label_from_name(filename: str) -> float:
    return common_data.parse_label_from_name(filename)


def read_signal(csv_path: Path):
    return common_data.read_signal(csv_path)


def normalize_model_type(model_type: str) -> str:
    return common_models.normalize_model_type(model_type)


def infer_lstm_layout_from_state_dict(state_dict):
    return common_models.infer_lstm_layout_from_state_dict(state_dict)


def build_model_from_config(model_cfg: dict, input_shape, state_dict=None):
    return common_models.build_model_from_config(model_cfg, input_shape, state_dict=state_dict)


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str = "raw",
):
    return common_data.build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode=feature_mode,
    )


def normalize_feature_mode(feature_mode: str) -> str:
    return common_data.normalize_feature_mode(feature_mode)


def apply_feature_mode(window: np.ndarray, feature_mode: str) -> np.ndarray:
    return common_data.apply_feature_mode(window, feature_mode)


def save_predictions(output_csv: Path, y_true: np.ndarray, y_pred: np.ndarray):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "prediction"])
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            writer.writerow([idx, float(t), float(p)])


def build_seg3_raw_aux_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
):
    """中文注释：为 seg3 构造未做 window_demean 的原始窗口序列，作为 aux/x_raw 输入。"""
    X_raw, y_raw = common_data.build_seg3_raw_aux_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
    )
    return X_raw, y_raw


def main():
    # 主流程保持“配置解析 -> 数据构建 -> 标准化 -> 模型推理 -> 保存结果”的顺序，
    # 方便快速定位到底是数据问题、配置问题还是模型问题。
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
    feature_mode = normalize_feature_mode(cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    max_samples = resolve_max_samples(args.max_samples, cfg.get("runtime", {}).get("max_samples", None))
    model_type = normalize_model_type(cfg["model"].get("type", "CNN-All"))

    t_total_start = time.perf_counter()
    X, y = build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode=feature_mode,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"No valid samples found under: {data_dir}")

    X_raw_aux = None
    if model_type == "cnn_tcn_seg3_soft_stats_moe":
        X_raw_aux, y_raw_aux = build_seg3_raw_aux_dataset(
            data_dir=data_dir,
            base_window_size=base_window_size,
            base_step=base_step,
            seq_length=seq_length,
            seq_step=seq_step,
        )
        if X_raw_aux.shape[0] == 0:
            raise RuntimeError(f"No raw aux samples found under: {data_dir}")
        if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y):
            raise RuntimeError("seg3 raw aux dataset is not aligned with main input dataset.")

    if max_samples is not None:
        limit = min(int(max_samples), int(X.shape[0]))
        X = X[:limit]
        y = y[:limit]
        if X_raw_aux is not None:
            X_raw_aux = X_raw_aux[:limit]

    X_scaled = common_scaler.apply_scaler(scaler_path, X)

    device = torch.device("cpu")
    state_dict = load_state_dict_compat(model_path, device)
    model = build_model_from_config(
        model_cfg=cfg["model"],
        input_shape=tuple(X_scaled.shape[1:]),
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X_scaled).to(device)
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        if model_type == "cnn_tcn_seg3_soft_stats_moe":
            aux_tensor = torch.from_numpy(X_raw_aux.astype(np.float32, copy=False)).to(device)
            if hasattr(model, "forward_debug"):
                prediction = model.forward_debug(X_tensor, aux=aux_tensor)["prediction"]
            else:
                outputs = model(X_tensor, aux=aux_tensor)
                prediction = outputs["prediction"] if isinstance(outputs, dict) else outputs
            y_pred = model.compose_prediction(prediction).cpu().numpy().reshape(-1)
        else:
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

    print("=== Raw Model Inference ===")
    print(f"data_dir: {data_dir}")
    print(f"model_path: {model_path}")
    print(f"scaler_path: {scaler_path}")
    print(f"model_type: {cfg['model'].get('type', 'CNN-All')}")
    print(f"feature_mode: {feature_mode}")
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
