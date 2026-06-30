"""PC 端可共享模型定义。

当前先收口 `infer.py` 和 `build_kmodel.py` 中已经审计为完全一致的模型：
CNN-LSTM、CNN-TCN 及其 TCN 基础块。CNNAll 和 Seg3 仍留在原脚本中，
等接口差异处理清楚后再迁移。
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_per_layer(value, num_layers: int, field: str):
    if isinstance(value, (list, tuple)):
        if len(value) != num_layers:
            raise ValueError(f"{field} length mismatch: {len(value)} vs {num_layers}")
        return list(value)
    return [value] * num_layers


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
        # 中文注释：与训练端保持一致，TCN 输出沿时间维做全局平均。
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

    def forward_debug(self, x, aux=None):
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

    def forward(self, x, aux=None):
        return self.forward_debug(x, aux=aux)["prediction"]

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


def build_cnn_all_from_config(model_cfg: dict, input_shape):
    conv_filters = model_cfg.get("conv_filters")
    if conv_filters is None:
        raise ValueError("CNN-All config missing conv filter definition.")
    return CNNAll(
        input_shape=input_shape,
        conv_filters=conv_filters,
        kernel_size=model_cfg.get("kernel_size", 3),
        pool_size=model_cfg.get("pool_size", 2),
    )


def build_cnn_tcn_from_config(model_cfg: dict, input_shape):
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


def build_cnn_tcn_seg3_from_config(model_cfg: dict, input_shape):
    prefix = "cnn_tcn_seg3_soft_stats_moe"
    conv_filters = model_cfg.get(f"{prefix}_conv_filters", model_cfg.get("cnn_tcn_conv_filters"))
    kernel_size = model_cfg.get(f"{prefix}_kernel_size", model_cfg.get("cnn_tcn_kernel_size", 3))
    pool_size = model_cfg.get(f"{prefix}_pool_size", model_cfg.get("cnn_tcn_pool_size", 2))
    tcn_num_channels = model_cfg.get(f"{prefix}_num_channels", model_cfg.get("cnn_tcn_num_channels"))
    tcn_kernel_size = model_cfg.get(f"{prefix}_tcn_kernel_size", 3)
    tcn_dilations = model_cfg.get(f"{prefix}_dilations")
    tcn_dropout = model_cfg.get(f"{prefix}_dropout", 0.0)
    if conv_filters is None:
        raise ValueError("Seg3 config missing conv filter definition.")
    if tcn_num_channels is None:
        raise ValueError("Seg3 config missing tcn channel definition.")
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


def build_cnn_lstm_from_config(model_cfg: dict, input_shape, state_dict=None):
    conv_filters = model_cfg.get("cnn_lstm_conv_filters", model_cfg.get("conv_filters"))
    kernel_size = model_cfg.get("cnn_lstm_kernel_size", model_cfg.get("kernel_size", 3))
    pool_size = model_cfg.get("cnn_lstm_pool_size", model_cfg.get("pool_size", 2))
    lstm_hidden_size = model_cfg.get("lstm_hidden_size", model_cfg.get("cnn_lstm_units"))
    lstm_num_layers = model_cfg.get("lstm_num_layers", 1)
    lstm_bidirectional = bool(model_cfg.get("lstm_bidirectional", False))
    if state_dict is not None:
        # 中文注释：以权重里的 LSTM 形状为准，避免配置残留字段构造错模型。
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


def build_shared_model_from_config(model_cfg: dict, input_shape, state_dict=None):
    """中文注释：只构建已经收口到公共层的模型；未收口模型返回 None。"""
    model_type = normalize_model_type(model_cfg.get("type", "CNN-All"))
    if model_type == "cnn_all":
        return build_cnn_all_from_config(model_cfg, input_shape)
    if model_type == "cnn_tcn":
        return build_cnn_tcn_from_config(model_cfg, input_shape)
    if model_type == "cnn_lstm":
        return build_cnn_lstm_from_config(model_cfg, input_shape, state_dict=state_dict)
    if model_type == "cnn_tcn_seg3_soft_stats_moe":
        return build_cnn_tcn_seg3_from_config(model_cfg, input_shape)
    return None


def build_model_from_config(model_cfg: dict, input_shape, state_dict=None):
    model = build_shared_model_from_config(model_cfg, input_shape, state_dict=state_dict)
    if model is None:
        raise ValueError(f"Unsupported model.type: {model_cfg.get('type', 'CNN-All')}")
    return model


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
