import argparse
import json
import os
import re
import site
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PC 端导出 K230 部署产物的主脚本。

这个脚本负责：
1. 读取 `k230_export_config.json`。
2. 从校准数据目录构建样本，并做标准化。
3. 导出 ONNX。
4. 导出给 K230 用的 `scaler json`。
5. 选择量化校准样本并调用 nncase 编译 `kmodel`。
"""


DEFAULT_CONFIG_PATH = "configs/export/k230_export_config_cnn_tcn.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build K230 deploy assets for Raw+CNN.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    # 大将军平时直接改配置文件即可；这里只保留一个临时覆盖入口，
    # 方便偶尔快速验证另一批校准数据，不影响默认配置用法。
    parser.add_argument(
        "--calibration_data_dir",
        type=str,
        default=None,
        help="Optional override for calibration data directory. If omitted, use paths.calibration_data_dir or paths.test_data_dir in config.",
    )
    parser.add_argument(
        "--skip_compile",
        action="store_true",
        help="Only export ONNX/scaler/calibration; skip nncase compile.",
    )
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=None,
        help="Override quantization.samples_count in config.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def resolve_calibration_sample_count(value, total: int, field_name: str) -> int:
    # 量化校准样本数支持两种配置：
    # 1. 正整数：表示最终参与量化校准的样本条数
    # 2. null   ：表示直接使用当前候选样本全集做全量校准
    if total <= 0:
        raise RuntimeError("No scaled samples available for calibration.")
    if value is None:
        return int(total)
    return min(require_positive_int(value, field_name), int(total))


def resolve_positive_step(value, fallback: int, field_name: str) -> int:
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


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
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
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


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str = "raw",
):
    # 这里把“目录里的原始长 CSV”转换成“模型真正吃的样本张量”。
    # 顺序是：
    # 1. 按文件名字典序遍历所有 CSV
    # 2. 逐个 CSV 按 base_window_size / base_step 切基础窗口
    # 3. 再按 sequence_length / sequence_step 组装成模型输入样本
    # 4. 最后把所有文件的样本顺序拼接成一个大数组
    X_list = []
    y_list = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = read_signal(csv_file)
        if signal.size < base_window_size:
            continue
        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            window = apply_feature_mode(window.astype(np.float32), feature_mode)
            features.append(window)
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


def build_seg3_raw_aux_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
):
    """中文注释：为 seg3 构造未做 window_demean 的原始窗口序列，作为 aux/x_raw 输入。"""
    return build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode="raw",
    )


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
        # 中文注释：先去窗口均值，再除以均值绝对值，突出相对基线的波动幅度。
        mean_value = float(np.mean(window, dtype=np.float32))
        denom = max(abs(mean_value), 1e-6)
        return ((window - mean_value) / denom).astype(np.float32)
    return window.astype(np.float32)


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
        for out_channels, k in zip(conv_filters, kernel_sizes):
            k = int(k)
            self.convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            in_channels = int(out_channels)
        self.pool_sizes = [int(p) for p in pool_sizes]
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(p) if p > 1 else nn.Identity() for p in self.pool_sizes]
        )

        length_after = self.features
        for p in self.pool_sizes:
            length_after = length_after // p
        flatten_dim = int(conv_filters[-1]) * int(length_after)
        self.fc = nn.Linear(flatten_dim, 1)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_filters,
        kernel_size=3,
        pool_size=2,
        lstm_hidden_size=50,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        lstm_bidirectional=False,
    ):
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

        in_channels = 1
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        length_after = self.features
        for out_channels, k, p in zip(conv_filters, kernel_sizes, pool_sizes):
            k = int(k)
            p = int(p)
            self.convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            self.pools.append(nn.MaxPool1d(p) if p > 1 else nn.Identity())
            in_channels = int(out_channels)
            length_after = length_after // p

        flatten_dim = int(conv_filters[-1]) * int(length_after)
        lstm_hidden_size = int(lstm_hidden_size)
        lstm_num_layers = int(lstm_num_layers)
        self.lstm = nn.LSTM(
            input_size=flatten_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=float(lstm_dropout) if lstm_num_layers > 1 else 0.0,
            bidirectional=bool(lstm_bidirectional),
            batch_first=True,
        )
        output_dim = lstm_hidden_size * (2 if bool(lstm_bidirectional) else 1)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, x):
        batch_size, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * time_steps, 1, feature_dim)
        for conv, pool in zip(self.convs, self.pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.downsample = (
            nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
            if int(in_channels) != int(out_channels)
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class CNNTCN(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_filters,
        kernel_size,
        pool_size,
        tcn_num_channels,
        tcn_kernel_size=3,
        tcn_dilations=None,
        tcn_dropout=0.0,
    ):
        super().__init__()
        time_steps, features = input_shape
        self.time_steps = int(time_steps)
        self.features = int(features)

        conv_filters = list(conv_filters)
        if not conv_filters:
            raise ValueError("cnn_tcn conv_filters must not be empty.")
        num_conv_layers = len(conv_filters)
        kernel_sizes = ensure_per_layer(kernel_size, num_conv_layers, "cnn_tcn.kernel_size")
        pool_sizes = ensure_per_layer(pool_size, num_conv_layers, "cnn_tcn.pool_size")

        in_channels = 1
        self.window_convs = nn.ModuleList()
        self.window_pools = nn.ModuleList()
        length_after = self.features
        for out_channels, k, p in zip(conv_filters, kernel_sizes, pool_sizes):
            k = int(k)
            p = int(p)
            self.window_convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            self.window_pools.append(nn.MaxPool1d(p) if p > 1 else nn.Identity())
            in_channels = int(out_channels)
            length_after = length_after // p

        window_feature_dim = int(conv_filters[-1])
        tcn_channels = [int(v) for v in tcn_num_channels]
        if not tcn_channels:
            raise ValueError("cnn_tcn tcn_num_channels must not be empty.")
        if tcn_dilations is None:
            tcn_dilations = [2 ** idx for idx in range(len(tcn_channels))]
        if len(tcn_dilations) != len(tcn_channels):
            raise ValueError("cnn_tcn tcn_dilations length mismatch with tcn_num_channels.")

        layers = []
        tcn_in_channels = window_feature_dim
        for out_channels, dilation in zip(tcn_channels, tcn_dilations):
            layers.append(
                TemporalBlock(
                    in_channels=tcn_in_channels,
                    out_channels=int(out_channels),
                    kernel_size=int(tcn_kernel_size),
                    dilation=int(dilation),
                    dropout=float(tcn_dropout),
                )
            )
            tcn_in_channels = int(out_channels)
        self.temporal_network = nn.Sequential(*layers)
        self.head = nn.Linear(tcn_in_channels, 1)

    def forward(self, x):
        batch_size, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * time_steps, 1, feature_dim)
        for conv, pool in zip(self.window_convs, self.window_pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.mean(dim=-1)
        x = x.reshape(batch_size, time_steps, -1).transpose(1, 2)
        x = self.temporal_network(x)
        # 这里必须和 VQ_Estimator 训练时的 CNN-TCN forward 保持一致：
        # TCN 输出沿时间维做全局平均，而不是取最后一个时间步。
        x = torch.mean(x, dim=2)
        return self.head(x)


class CNNTCNSeg3SoftStatsMoE(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_filters,
        kernel_size,
        pool_size,
        tcn_num_channels,
        tcn_kernel_size=3,
        tcn_dilations=None,
        tcn_dropout=0.0,
        stats_hidden_dim=16,
        router_hidden_dim=24,
        expert_hidden_dim=32,
        route_temperature=1.0,
        seg3_min_value=0.0,
        seg3_max_value=1.5,
        low_expert_upper=0.4,
        mid_expert_lower=0.2,
        mid_expert_upper=1.0,
        high_expert_lower=0.8,
    ):
        super().__init__()
        time_steps, features = input_shape
        self.time_steps = int(time_steps)
        self.features = int(features)
        self.route_temperature = max(float(route_temperature), 1e-6)
        self.seg3_min_value = float(seg3_min_value)
        self.seg3_max_value = float(seg3_max_value)
        self.low_expert_upper = float(low_expert_upper)
        self.mid_expert_lower = float(mid_expert_lower)
        self.mid_expert_upper = float(mid_expert_upper)
        self.high_expert_lower = float(high_expert_lower)

        conv_filters = list(conv_filters)
        if not conv_filters:
            raise ValueError("seg3 conv_filters must not be empty.")
        kernel_sizes = ensure_per_layer(kernel_size, len(conv_filters), "seg3.kernel_size")
        pool_sizes = ensure_per_layer(pool_size, len(conv_filters), "seg3.pool_size")

        in_channels = 1
        self.window_convs = nn.ModuleList()
        self.window_pools = nn.ModuleList()
        for out_channels, k, p in zip(conv_filters, kernel_sizes, pool_sizes):
            k = int(k)
            p = int(p)
            self.window_convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            self.window_pools.append(nn.MaxPool1d(p) if p > 1 else nn.Identity())
            in_channels = int(out_channels)

        self.stats_mlp = nn.Sequential(
            nn.Linear(4, int(stats_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(stats_hidden_dim), int(stats_hidden_dim)),
            nn.ReLU(),
        )
        self.stats_post_norm = nn.LayerNorm(int(stats_hidden_dim))

        tcn_channels = [int(v) for v in tcn_num_channels]
        if not tcn_channels:
            raise ValueError("seg3 tcn_num_channels must not be empty.")
        if tcn_dilations is None:
            tcn_dilations = [2 ** idx for idx in range(len(tcn_channels))]
        if len(tcn_dilations) != len(tcn_channels):
            raise ValueError("seg3 tcn_dilations length mismatch with tcn_num_channels.")

        layers = []
        tcn_in_channels = int(conv_filters[-1]) + int(stats_hidden_dim)
        for out_channels, dilation in zip(tcn_channels, tcn_dilations):
            layers.append(
                TemporalBlock(
                    in_channels=tcn_in_channels,
                    out_channels=int(out_channels),
                    kernel_size=int(tcn_kernel_size),
                    dilation=int(dilation),
                    dropout=float(tcn_dropout),
                )
            )
            tcn_in_channels = int(out_channels)
        self.temporal_network = nn.Sequential(*layers)

        head_dim = int(tcn_in_channels) * 2 + int(stats_hidden_dim) * 2 + 4
        self.head_input_norm = nn.LayerNorm(head_dim)
        self.route_head = nn.Sequential(
            nn.Linear(head_dim, int(router_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(router_hidden_dim), 3),
        )
        self.low_expert = self._make_expert(head_dim, int(expert_hidden_dim))
        self.mid_expert = self._make_expert(head_dim, int(expert_hidden_dim))
        self.high_expert = self._make_expert(head_dim, int(expert_hidden_dim))

    @staticmethod
    def _make_expert(input_dim: int, hidden_dim: int):
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _extract_stats_features(self, x_raw):
        mean = x_raw.mean(dim=2)
        std = x_raw.std(dim=2, unbiased=False)
        value_range = x_raw.amax(dim=2) - x_raw.amin(dim=2)
        rms = torch.sqrt(torch.mean(x_raw * x_raw, dim=2) + 1e-8)
        stats = torch.stack([mean, std, value_range, rms], dim=2)
        shortcut = torch.stack([mean, rms], dim=2)
        stats = torch.sign(stats) * torch.log1p(torch.abs(stats))
        shortcut = torch.sign(shortcut) * torch.log1p(torch.abs(shortcut))
        return stats, shortcut

    def _bounded_expert(self, raw, lower: float, upper: float):
        return float(lower) + torch.sigmoid(raw) * (float(upper) - float(lower))

    def forward(self, x, aux=None):
        if aux is None:
            aux = x
        if aux.shape != x.shape:
            raise ValueError("seg3 aux input must have the same shape as main input.")

        batch_size, time_steps, feature_dim = x.shape
        stats_all, shortcut_mean_rms = self._extract_stats_features(aux)
        stats_encoded = self.stats_post_norm(self.stats_mlp(stats_all))

        x_main = x.reshape(batch_size * time_steps, 1, feature_dim)
        for conv, pool in zip(self.window_convs, self.window_pools):
            x_main = torch.relu(conv(x_main))
            x_main = pool(x_main)
        x_main = x_main.mean(dim=-1).reshape(batch_size, time_steps, -1)

        fused = torch.cat([x_main, stats_encoded], dim=2).transpose(1, 2)
        temporal = self.temporal_network(fused)
        tcn_mean = temporal.mean(dim=2)
        tcn_max = temporal.amax(dim=2)
        stats_mean = stats_encoded.mean(dim=1)
        stats_max = stats_encoded.amax(dim=1)
        shortcut_mean = shortcut_mean_rms.mean(dim=1)
        shortcut_max = shortcut_mean_rms.amax(dim=1)
        head_input = torch.cat([tcn_mean, tcn_max, stats_mean, stats_max, shortcut_mean, shortcut_max], dim=1)
        head_input = self.head_input_norm(head_input)

        route_logits = self.route_head(head_input)
        route_probs = F.softmax(route_logits / self.route_temperature, dim=1)
        low_pred = self._bounded_expert(self.low_expert(head_input), self.seg3_min_value, self.low_expert_upper)
        mid_pred = self._bounded_expert(self.mid_expert(head_input), self.mid_expert_lower, self.mid_expert_upper)
        high_pred = self._bounded_expert(self.high_expert(head_input), self.high_expert_lower, self.seg3_max_value)
        expert_preds = torch.cat([low_pred, mid_pred, high_pred], dim=1)
        prediction = torch.sum(route_probs * expert_preds, dim=1, keepdim=True)
        return prediction

    def compose_prediction(self, prediction):
        return torch.clamp(prediction.squeeze(1), min=self.seg3_min_value, max=self.seg3_max_value)


class Seg3ExportWrapper(nn.Module):
    def __init__(self, model: CNNTCNSeg3SoftStatsMoE):
        super().__init__()
        self.model = model

    def forward(self, x, raw_input):
        prediction = self.model(x, aux=raw_input)
        return self.model.compose_prediction(prediction).unsqueeze(1)


def normalize_model_type(model_type: str) -> str:
    text = str(model_type).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"cnn_all", "cnn"}:
        return "cnn_all"
    if text in {"cnn_lstm", "cnnlstm"}:
        return "cnn_lstm"
    if text in {"cnn_tcn", "cnntcn"}:
        return "cnn_tcn"
    if text in {"cnn_tcn_seg3_soft_stats_moe", "cnn_tcn_seg3", "cnntcnseg3"}:
        return "cnn_tcn_seg3_soft_stats_moe"
    raise ValueError(f"Unsupported model.type: {model_type}")


def infer_lstm_layout_from_state_dict(state_dict):
    hidden_size = None
    num_layers = None
    bidirectional = False
    layers = set()
    pattern = re.compile(r"^lstm\.weight_ih_l(\d+)(_reverse)?$")
    for key, value in state_dict.items():
        match = pattern.match(str(key))
        if not match:
            continue
        layer_idx = int(match.group(1))
        layers.add(layer_idx)
        if match.group(2):
            bidirectional = True
        if layer_idx == 0 and hidden_size is None:
            hidden_size = int(value.shape[0] // 4)
    if layers:
        num_layers = max(layers) + 1
    return hidden_size, num_layers, bidirectional


def build_model_from_config(model_cfg: dict, input_shape, state_dict=None):
    model_type = normalize_model_type(model_cfg.get("type", "CNN-All"))
    if model_type == "cnn_all":
        return CNNAll(
            input_shape=input_shape,
            conv_filters=model_cfg["conv_filters"],
            kernel_size=model_cfg["kernel_size"],
            pool_size=model_cfg["pool_size"],
        )
    if model_type == "cnn_tcn":
        conv_filters = model_cfg.get("cnn_tcn_conv_filters", model_cfg.get("conv_filters"))
        kernel_size = model_cfg.get("cnn_tcn_kernel_size", model_cfg.get("kernel_size", 3))
        pool_size = model_cfg.get("cnn_tcn_pool_size", model_cfg.get("pool_size", 2))
        tcn_num_channels = model_cfg.get("cnn_tcn_num_channels", model_cfg.get("tcn_num_channels"))
        tcn_kernel_size = model_cfg.get("cnn_tcn_tcn_kernel_size", model_cfg.get("tcn_kernel_size", 3))
        tcn_dilations = model_cfg.get("cnn_tcn_dilations", model_cfg.get("tcn_dilations"))
        tcn_dropout = model_cfg.get("cnn_tcn_dropout", model_cfg.get("tcn_dropout", 0.0))
        if conv_filters is None:
            raise ValueError("CNN-TCN config missing conv filter definition.")
        if tcn_num_channels is None:
            raise ValueError("CNN-TCN config missing tcn channel definition.")
        return CNNTCN(
            input_shape=input_shape,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            tcn_num_channels=tcn_num_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
            tcn_dropout=tcn_dropout,
        )
    if model_type == "cnn_tcn_seg3_soft_stats_moe":
        prefix = "cnn_tcn_seg3_soft_stats_moe"
        conv_filters = model_cfg.get(f"{prefix}_conv_filters", model_cfg.get("conv_filters"))
        kernel_size = model_cfg.get(f"{prefix}_kernel_size", model_cfg.get("kernel_size", 3))
        pool_size = model_cfg.get(f"{prefix}_pool_size", model_cfg.get("pool_size", 2))
        tcn_num_channels = model_cfg.get(f"{prefix}_num_channels", model_cfg.get("tcn_num_channels"))
        tcn_kernel_size = model_cfg.get(f"{prefix}_tcn_kernel_size", model_cfg.get("tcn_kernel_size", 3))
        tcn_dilations = model_cfg.get(f"{prefix}_dilations", model_cfg.get("tcn_dilations"))
        tcn_dropout = model_cfg.get(f"{prefix}_dropout", model_cfg.get("tcn_dropout", 0.0))
        if conv_filters is None:
            raise ValueError("CNN-TCN-Seg3 config missing conv filter definition.")
        if tcn_num_channels is None:
            raise ValueError("CNN-TCN-Seg3 config missing tcn channel definition.")
        return CNNTCNSeg3SoftStatsMoE(
            input_shape=input_shape,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            tcn_num_channels=tcn_num_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
            tcn_dropout=tcn_dropout,
            stats_hidden_dim=model_cfg.get(f"{prefix}_stats_hidden_dim", 16),
            router_hidden_dim=model_cfg.get(f"{prefix}_router_hidden_dim", 24),
            expert_hidden_dim=model_cfg.get(f"{prefix}_expert_hidden_dim", 32),
            route_temperature=model_cfg.get(f"{prefix}_route_temperature", 1.0),
            seg3_min_value=model_cfg.get(f"{prefix}_min_value", 0.0),
            seg3_max_value=model_cfg.get(f"{prefix}_max_value", 1.5),
            low_expert_upper=model_cfg.get(f"{prefix}_low_expert_upper", 0.4),
            mid_expert_lower=model_cfg.get(f"{prefix}_mid_expert_lower", 0.2),
            mid_expert_upper=model_cfg.get(f"{prefix}_mid_expert_upper", 1.0),
            high_expert_lower=model_cfg.get(f"{prefix}_high_expert_lower", 0.8),
        )

    conv_filters = model_cfg.get("cnn_lstm_conv_filters", model_cfg.get("conv_filters"))
    kernel_size = model_cfg.get("cnn_lstm_kernel_size", model_cfg.get("kernel_size", 3))
    pool_size = model_cfg.get("cnn_lstm_pool_size", model_cfg.get("pool_size", 2))
    lstm_hidden_size = model_cfg.get("lstm_hidden_size", model_cfg.get("cnn_lstm_units"))
    lstm_num_layers = model_cfg.get("lstm_num_layers", 1)
    lstm_bidirectional = bool(model_cfg.get("lstm_bidirectional", False))
    if state_dict is not None:
        # 以权重里的 LSTM 形状为准，避免配置里残留的隐藏维度写错。
        inferred_hidden_size, inferred_num_layers, inferred_bidirectional = infer_lstm_layout_from_state_dict(state_dict)
        if inferred_hidden_size is not None:
            lstm_hidden_size = inferred_hidden_size
        if inferred_num_layers is not None:
            lstm_num_layers = inferred_num_layers
        if inferred_bidirectional:
            lstm_bidirectional = True
    if conv_filters is None:
        raise ValueError("CNN-LSTM config missing conv filter definition.")
    if lstm_hidden_size is None:
        raise ValueError("CNN-LSTM config missing lstm hidden size.")
    return CNNLSTM(
        input_shape=input_shape,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=model_cfg.get("lstm_dropout", 0.0),
        lstm_bidirectional=lstm_bidirectional,
    )


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def export_onnx(model: nn.Module, onnx_path: Path, input_shape, raw_input_shape=None):
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires `onnx` package. Install requirements_k230_host.txt first."
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, *input_shape, dtype=torch.float32)
    if raw_input_shape is not None:
        raw_dummy = torch.randn(1, *raw_input_shape, dtype=torch.float32)
        export_model = Seg3ExportWrapper(model)
        torch.onnx.export(
            export_model,
            (dummy, raw_dummy),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input", "raw_input"],
            output_names=["output"],
            dynamic_axes=None,
        )
        sanitize_onnx_for_nncase(onnx_path, onnx)
        return
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    sanitize_onnx_for_nncase(onnx_path, onnx)


def sanitize_onnx_for_nncase(onnx_path: Path, onnx_module):
    # 某些 1D MaxPool 导出的 ONNX 会附带 dilations=[1] 属性，
    # 但当前 nncase 在导入时会把它当成尺寸错误的窗口参数。
    # 这里在导出后做一次轻量清洗，移除这项对结果无影响的属性。
    model = onnx_module.load(onnx_path.as_posix())
    changed = False
    for node in model.graph.node:
        if node.op_type != "MaxPool":
            continue
        keep_attrs = []
        for attr in node.attribute:
            if attr.name == "dilations":
                changed = True
                continue
            keep_attrs.append(attr)
        if len(keep_attrs) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(keep_attrs)
    if changed:
        onnx_module.save(model, onnx_path.as_posix())


def export_scaler_json(scaler_pkl: Path, scaler_json: Path):
    # K230 板端不能直接读取 sklearn 的 pkl，所以这里额外导出一份纯 json 参数。
    scaler = joblib.load(scaler_pkl)
    payload = {
        "type": "StandardScaler",
        "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
        "mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
    }
    save_json(scaler_json, payload)


def apply_scaler(scaler_pkl: Path, X: np.ndarray):
    scaler = joblib.load(scaler_pkl)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape).astype(np.float32)
    return X_scaled


def normalize_sampling_strategy(strategy: str) -> str:
    # 量化校准样本抽取策略统一在这里标准化，配置里写法可以更宽松。
    # first   : 直接取前 N 条样本
    # uniform : 从全体样本中均匀抽 N 条
    # random  : 从全体样本中随机抽 N 条
    # per_dryness_uniform : 先按干度分组，再尽量给每个干度均匀分配样本配额
    text = str(strategy).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"first", "head", "sequential"}:
        return "first"
    if text in {"uniform", "even", "linspace"}:
        return "uniform"
    if text in {"random", "shuffle", "rand"}:
        return "random"
    if text in {"per_dryness_uniform", "by_dryness", "stratified", "stratified_uniform"}:
        return "per_dryness_uniform"
    raise ValueError(
        "quantization.sampling_strategy must be one of: first / uniform / random / per_dryness_uniform"
    )


def validate_quantization_config(qcfg: dict):
    # K230 当前这条 nncase/PTQ 路径里，量化类型可写 uint8 / int8 / int16，
    # 但激活量化和权重量化不能同时都写成 int16。
    quant_type = str(qcfg.get("quant_type", "uint8")).strip().lower()
    weight_quant_type = str(qcfg.get("weight_quant_type", "uint8")).strip().lower()
    allowed = {"uint8", "int8", "int16"}
    if quant_type not in allowed:
        raise ValueError(
            "quantization.quant_type must be one of: uint8 / int8 / int16"
        )
    if weight_quant_type not in allowed:
        raise ValueError(
            "quantization.weight_quant_type must be one of: uint8 / int8 / int16"
        )
    if quant_type == "int16" and weight_quant_type == "int16":
        raise ValueError(
            "K230 当前量化配置里，quantization.quant_type 和 "
            "quantization.weight_quant_type 不能同时都是 int16。"
        )


def prepare_nncase_env():
    # 某些环境里即使已经安装 nncase / nncase-kpu，如果没有补上
    # NNCASE_PLUGIN_PATH 和 site-packages 到 PATH，K230 相关插件也可能没有被正确加载。
    # 结果就是：脚本表面能编译出 kmodel，但导出的 K230 产物可能不正常。
    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_roots.append(user_site)
    except Exception:
        pass

    unique_roots = []
    seen = set()
    for raw in site_roots:
        if not raw:
            continue
        norm = str(Path(raw).resolve())
        if norm in seen:
            continue
        seen.add(norm)
        unique_roots.append(Path(norm))

    plugin_dir = None
    package_root = None
    for root in unique_roots:
        candidate = root / "nncase" / "modules" / "kpu"
        if candidate.exists():
            plugin_dir = candidate
            package_root = root
            break

    if plugin_dir is not None and not os.environ.get("NNCASE_PLUGIN_PATH"):
        os.environ["NNCASE_PLUGIN_PATH"] = plugin_dir.as_posix()

    if package_root is not None:
        current_path = os.environ.get("PATH", "")
        package_root_text = str(package_root)
        path_items = current_path.split(os.pathsep) if current_path else []
        if package_root_text not in path_items:
            os.environ["PATH"] = package_root_text + os.pathsep + current_path if current_path else package_root_text


def _select_per_dryness_uniform_indices(y_labels: np.ndarray, count: int) -> np.ndarray:
    # 大将军，这里实现“每个干度尽量都取到一些样本”的分层抽样。
    # 做法是：
    # 1. 先按干度标签分组
    # 2. 先给每个干度分一个基础配额
    # 3. 剩余名额再按各组剩余容量继续分配
    # 4. 组内用 linspace 均匀取点，避免只拿到每组头部样本
    unique_labels = []
    grouped_indices = []
    for value in sorted({float(v) for v in y_labels.tolist()}):
        group = np.flatnonzero(y_labels == np.float32(value)).astype(np.int64)
        if group.size == 0:
            continue
        unique_labels.append(value)
        grouped_indices.append(group)

    if not grouped_indices:
        return np.empty((0,), dtype=np.int64)

    group_count = len(grouped_indices)
    base_quota = max(1, count // group_count)
    selected_parts = []
    used = 0
    remaining_capacity = []

    for group in grouped_indices:
        take = min(base_quota, int(group.size))
        if take > 0:
            picks = np.linspace(0, int(group.size) - 1, num=take, dtype=np.int64)
            selected_parts.append(group[picks])
            used += take
        remaining_capacity.append(max(0, int(group.size) - take))

    rest = max(0, int(count) - int(used))
    if rest > 0:
        while rest > 0:
            progressed = False
            for idx, group in enumerate(grouped_indices):
                if remaining_capacity[idx] <= 0:
                    continue
                already = int(group.size) - int(remaining_capacity[idx])
                # 在剩余区间内继续均匀补点，保持组内覆盖尽量分散
                offset_candidates = np.arange(already, int(group.size), dtype=np.int64)
                if offset_candidates.size <= 0:
                    remaining_capacity[idx] = 0
                    continue
                pick_pos = np.linspace(0, int(offset_candidates.size) - 1, num=1, dtype=np.int64)[0]
                selected_parts.append(group[offset_candidates[pick_pos:pick_pos + 1]])
                remaining_capacity[idx] -= 1
                rest -= 1
                progressed = True
                if rest <= 0:
                    break
            if not progressed:
                break

    if not selected_parts:
        return np.empty((0,), dtype=np.int64)

    indices = np.concatenate(selected_parts, axis=0).astype(np.int64)
    indices = np.unique(indices)
    if indices.size > count:
        indices = indices[:count]
    return np.sort(indices)


def select_calibration_data(X_scaled: np.ndarray, count: int, strategy: str, random_seed, y_labels=None):
    # 量化校准真正喂给 nncase 的样本在这里确定。
    # 大将军以后如果想调整“拿哪些样本去量化”，只需要改
    # k230_export_config.json 里的：
    # - quantization.samples_count
    # - quantization.sampling_strategy
    # - quantization.random_seed
    total = int(X_scaled.shape[0])
    if total <= 0:
        raise RuntimeError("No scaled samples available for calibration.")
    if count >= total:
        return X_scaled.astype(np.float32)

    mode = normalize_sampling_strategy(strategy)
    if mode == "first":
        # 保留旧逻辑：按当前样本顺序直接取前 N 条。
        indices = np.arange(count, dtype=np.int64)
    elif mode == "uniform":
        # 从整体样本范围内均匀取点，适合尽量覆盖全局分布。
        indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
    elif mode == "per_dryness_uniform":
        if y_labels is None or int(len(y_labels)) != total:
            raise ValueError(
                "sampling_strategy=per_dryness_uniform requires y_labels with same length as X_scaled"
            )
        indices = _select_per_dryness_uniform_indices(np.asarray(y_labels, dtype=np.float32), int(count))
        if indices.size == 0:
            raise RuntimeError("No calibration indices selected for per_dryness_uniform.")
    else:
        # 随机抽样适合做对照实验；random_seed 固定时结果可复现。
        if random_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(random_seed))
        indices = np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))

    return X_scaled[indices].astype(np.float32)


def compile_kmodel_with_nncase(cfg: dict, root: Path, calibration_data: np.ndarray, raw_aux_data=None):
    prepare_nncase_env()
    try:
        import nncase  # type: ignore
    except ImportError as exc:
        raise RuntimeError("nncase is not installed in current environment.") from exc

    paths = cfg["paths"]
    qcfg = cfg["quantization"]
    debug_cfg = cfg.get("debug", {})
    validate_quantization_config(qcfg)
    onnx_path = (root / paths["onnx"]).resolve()
    kmodel_path = (root / paths["kmodel"]).resolve()
    dump_dir = (root / paths["nncase_dump_dir"]).resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    kmodel_path.parent.mkdir(parents=True, exist_ok=True)

    with onnx_path.open("rb") as f:
        model_content = f.read()

    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.preprocess = False
    compile_options.dump_ir = bool(debug_cfg.get("dump_ir", False))
    compile_options.dump_asm = bool(debug_cfg.get("dump_asm", False))
    compile_options.dump_dir = dump_dir.as_posix()

    ptq_options = nncase.PTQTensorOptions()
    # 下面这些字段就是量化的核心配置：
    # samples_count      : 最终实际参与量化校准的样本条数
    # quant_type         : 激活值量化类型
    # weight_quant_type  : 权重量化类型
    # calibrate_method   : 量化范围估计方法
    ptq_options.samples_count = int(calibration_data.shape[0])
    ptq_options.quant_type = qcfg.get("quant_type", "uint8")
    ptq_options.w_quant_type = qcfg.get("weight_quant_type", "uint8")
    ptq_options.calibrate_method = qcfg.get("calibrate_method", "NoClip")

    # nncase 这里吃的是“样本列表”，每条样本 shape 都是 (1, C, L)。
    sample_list = [calibration_data[i : i + 1].astype(np.float32) for i in range(calibration_data.shape[0])]
    if raw_aux_data is not None:
        raw_sample_list = [raw_aux_data[i : i + 1].astype(np.float32) for i in range(raw_aux_data.shape[0])]
        ptq_options.set_tensor_data([sample_list, raw_sample_list])
    else:
        ptq_options.set_tensor_data([sample_list])

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(model_content, import_options)
    compiler.use_ptq(ptq_options)
    compiler.compile()
    kmodel = compiler.gencode_tobytes()

    with kmodel_path.open("wb") as f:
        f.write(kmodel)


def main():
    # 主流程里会显式打印这次导出的关键产物，方便确认实际生成的是哪一版文件。
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_json(cfg_path)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    qcfg = cfg["quantization"]
    feature_mode = normalize_feature_mode(cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    model_type = normalize_model_type(model_cfg.get("type", "CNN-All"))

    weights_pth = (root / paths["weights_pth"]).resolve()
    onnx_path = (root / paths["onnx"]).resolve()
    scaler_pkl = (root / paths["scaler_pkl"]).resolve()
    scaler_json = (root / paths["scaler_json"]).resolve()
    calib_npy = (root / paths["calibration_npy"]).resolve()
    if args.calibration_data_dir:
        # 如果临时传了目录，就优先用临时目录。
        calibration_data_dir = Path(args.calibration_data_dir).resolve()
    else:
        # 正常长期使用时，大将军只需要改配置里的 calibration_data_dir。
        # 如果没写这个字段，才回退到旧字段 test_data_dir。
        calib_data_dir_cfg = paths.get("calibration_data_dir", paths["test_data_dir"])
        calibration_data_dir = (root / calib_data_dir_cfg).resolve()

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    try:
        X, y = build_dataset(
            data_dir=calibration_data_dir,
            base_window_size=base_window,
            base_step=base_step,
            seq_length=seq_length,
            seq_step=seq_step,
            feature_mode=feature_mode,
        )
        if X.shape[0] == 0:
            raise RuntimeError(f"No valid samples in calibration data: {calibration_data_dir}")

        X_raw_aux = None
        if model_type == "cnn_tcn_seg3_soft_stats_moe":
            X_raw_aux, y_raw_aux = build_seg3_raw_aux_dataset(
                data_dir=calibration_data_dir,
                base_window_size=base_window,
                base_step=base_step,
                seq_length=seq_length,
                seq_step=seq_step,
            )
            if X_raw_aux.shape[0] == 0:
                raise RuntimeError(f"No raw aux samples in calibration data: {calibration_data_dir}")
            if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y):
                raise RuntimeError("seg3 raw aux calibration dataset is not aligned with main calibration dataset.")
        X_scaled = apply_scaler(scaler_pkl, X)
        if args.max_calib_samples is not None:
            count = resolve_calibration_sample_count(
                args.max_calib_samples,
                X_scaled.shape[0],
                "max_calib_samples",
            )
        else:
            # 平时优先改配置里的 quantization.samples_count：
            # - 填正整数：表示量化校准最终使用多少条样本
            # - 填 null  ：表示直接用当前候选样本全集做全量校准
            count = resolve_calibration_sample_count(
                qcfg.get("samples_count", 64),
                X_scaled.shape[0],
                "quantization.samples_count",
            )
        sampling_strategy = qcfg.get("sampling_strategy", "first")
        random_seed = qcfg.get("random_seed", None)
        # 这里是“从全部候选样本里抽出量化校准子集”的唯一入口。
        calibration_data = select_calibration_data(
            X_scaled=X_scaled,
            count=count,
            strategy=sampling_strategy,
            random_seed=random_seed,
            y_labels=y,
        )
        raw_aux_data = None
        if model_type == "cnn_tcn_seg3_soft_stats_moe":
            raw_aux_data = select_calibration_data(
                X_scaled=X_raw_aux,
                count=count,
                strategy=sampling_strategy,
                random_seed=random_seed,
                y_labels=y,
            )
        calib_npy.parent.mkdir(parents=True, exist_ok=True)
        # 额外保存一份 calibration_input.npy，方便后续排查到底用了哪批样本去量化。
        np.save(calib_npy, calibration_data)

        export_scaler_json(scaler_pkl, scaler_json)

        input_shape = tuple(X_scaled.shape[1:])
        state_dict = load_state_dict_compat(weights_pth, torch.device("cpu"))
        model = build_model_from_config(
            model_cfg=model_cfg,
            input_shape=input_shape,
            state_dict=state_dict,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        raw_input_shape = input_shape if model_type == "cnn_tcn_seg3_soft_stats_moe" else None
        export_onnx(model, onnx_path, input_shape=input_shape, raw_input_shape=raw_input_shape)

        print("Exported ONNX:", onnx_path)
        print("Exported scaler json:", scaler_json)
        print("Calibration data dir:", calibration_data_dir)
        print("Saved calibration data:", calib_npy, calibration_data.shape)
        print("Model type:", model_cfg.get("type", "CNN-All"))
        print("Calibration sampling strategy:", normalize_sampling_strategy(sampling_strategy))
        print("Calibration random seed:", random_seed)
        print("feature_mode:", feature_mode)

        if args.skip_compile:
            print("Skip nncase compile (--skip_compile set).")
            return

        compile_kmodel_with_nncase(cfg, root, calibration_data, raw_aux_data=raw_aux_data)
        print("Generated kmodel:", (root / paths["kmodel"]).resolve())
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
