"""PC 端 CSV 切窗与特征处理公共工具。"""

from pathlib import Path

import numpy as np


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
    values = []
    with Path(csv_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",")[0].strip()
            try:
                values.append(float(first))
            except ValueError:
                continue
    return np.asarray(values, dtype=np.float32)


def normalize_feature_mode(feature_mode: str) -> str:
    text = str(feature_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"window_demean", "demean", "window_mean_center"}:
        return "window_demean"
    if text in {"window_rel_demean", "relative_demean", "window_mean_ratio"}:
        return "window_rel_demean"
    return "raw"


def apply_feature_mode(window: np.ndarray, feature_mode: str) -> np.ndarray:
    mode = normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        return (window - np.mean(window, dtype=np.float32)).astype(np.float32)
    if mode == "window_rel_demean":
        # 中文注释：先去窗口均值，再按均值绝对值归一，突出相对基线波动。
        mean_value = float(np.mean(window, dtype=np.float32))
        denom = max(abs(mean_value), 1e-6)
        return ((window - mean_value) / denom).astype(np.float32)
    return window.astype(np.float32)


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str = "raw",
):
    # 中文注释：把目录中的原始长 CSV 转成模型输入样本，供推理和导出共用。
    X_list = []
    y_list = []
    for csv_file in sorted(Path(data_dir).glob("*.csv")):
        signal = read_signal(csv_file)
        if signal.size < base_window_size:
            continue
        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            features.append(apply_feature_mode(window.astype(np.float32), feature_mode))
        if len(features) < seq_length:
            continue
        label = parse_label_from_name(csv_file.name)
        for i in range(0, len(features) - seq_length + 1, seq_step):
            X_list.append(np.stack(features[i : i + seq_length], axis=0))
            y_list.append(label)
    if not X_list:
        return np.empty((0, seq_length, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list).astype(np.float32), np.asarray(y_list, dtype=np.float32)


def build_seg3_raw_aux_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
):
    """中文注释：为 seg3 构造未做特征变换的原始窗口序列，作为 aux/x_raw 输入。"""
    return build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode="raw",
    )
