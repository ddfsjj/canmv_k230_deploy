import argparse
import csv
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def resolve_max_samples(cli_value, config_value):
    # 优先使用命令行；未传时再读取配置。配置为 null 表示全量。
    if cli_value is not None:
        value = int(cli_value)
        if value <= 0:
            raise ValueError(f"max_samples must be > 0, got {value}")
        return value
    if config_value is None:
        return None
    value = int(config_value)
    if value <= 0:
        raise ValueError(f"runtime.max_samples must be > 0, got {value}")
    return value


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
        return {
            "prediction": prediction,
            "route_logits": route_logits,
            "route_probs": route_probs,
            "expert_preds": expert_preds,
            "low_pred": low_pred,
            "mid_pred": mid_pred,
            "high_pred": high_pred,
            "head_input": head_input,
        }

    def compose_prediction(self, prediction):
        return torch.clamp(prediction.squeeze(1), min=self.seg3_min_value, max=self.seg3_max_value)


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
        # 以权重里的 LSTM 形状为准，避免 meta 残留字段把模型构造错。
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


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str = "raw",
):
    # 把目录中的原始长序列 CSV 切成模型真正使用的样本张量。
    # 这里的切窗顺序必须和导出脚本保持一致，否则后面的误差对比会错位。
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


def normalize_feature_mode(feature_mode: str) -> str:
    text = str(feature_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"window_demean", "demean", "window_mean_center"}:
        return "window_demean"
    return "raw"


def apply_feature_mode(window: np.ndarray, feature_mode: str) -> np.ndarray:
    mode = normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        return (window - np.mean(window, dtype=np.float32)).astype(np.float32)
    return window.astype(np.float32)


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
    X_raw, y_raw = build_dataset(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode="raw",
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

    scaler = joblib.load(scaler_path)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape).astype(np.float32)

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
            outputs = model(X_tensor, aux=aux_tensor)
            y_pred = model.compose_prediction(outputs["prediction"]).cpu().numpy().reshape(-1)
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
