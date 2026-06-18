import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

import compare_pth_kmodel
import infer


class LSTMLastFeatureExtractor(nn.Module):
    """澶у皢鍐涳紝杩欎釜鍖呰鍣ㄥ彧鎻愬彇 CNN-LSTM 鍦ㄦ渶鍚庝竴涓椂闂存鐨勭壒寰?Y[4]銆?""

    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        batch_size, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * time_steps, 1, feature_dim)
        for conv, pool in zip(self.model.convs, self.model.pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.model.lstm(x)
        return x[:, -1, :]


def parse_args():
    parser = argparse.ArgumentParser(description="Test regression-head sensitivity on PC float CNN-LSTM.")
    parser.add_argument(
        "--infer_config",
        type=str,
        default="configs/infer/infer_config_cnn_lstm_20260412_125022.json",
        help="推理配置路径。",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../raw_cnn_k230/generated_dry_temp_csv",
        help="数据目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/sensitivity/regression_head_20260412",
        help="输出目录。",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.75,
        help="高干度阈值，true_label >= 该值视为高干度。",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=5,
        help="每组抽取样本数。",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="对 Y[4] 全部 50 维统一加减的扰动幅度。",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "global_index",
        "sample_id",
        "source_csv",
        "true_label",
        "original_output",
        "plus_output",
        "minus_output",
        "plus_abs_change",
        "minus_abs_change",
        "plus_sensitivity",
        "minus_sensitivity",
        "y4_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_float_model(root: Path, infer_cfg: dict, input_shape):
    weights_path = (root / infer_cfg["model"]["weights_path"]).resolve()
    state_dict = infer.load_state_dict_compat(weights_path, torch.device("cpu"))
    model = infer.build_model_from_config(
        model_cfg=infer_cfg["model"],
        input_shape=input_shape,
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_dataset(root: Path, infer_cfg: dict, data_dir: Path):
    data_cfg = infer_cfg["data"]
    X, y, source = compare_pth_kmodel.build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=compare_pth_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=compare_pth_kmodel.require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=compare_pth_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=compare_pth_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw")),
        max_samples=None,
    )
    scaler_path = (root / infer_cfg["normalization"]["scaler_path"]).resolve()
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    return X_scaled, y.astype(np.float32), source


def evenly_pick_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if indices.size < count:
        raise ValueError(f"候选样本不足，要求 {count} 个，实际只有 {indices.size} 个。")
    positions = np.linspace(0, indices.size - 1, num=count, dtype=np.int64)
    return indices[positions]


def summarize_group(rows):
    original = np.asarray([row["original_output"] for row in rows], dtype=np.float32)
    plus_change = np.asarray([row["plus_abs_change"] for row in rows], dtype=np.float32)
    minus_change = np.asarray([row["minus_abs_change"] for row in rows], dtype=np.float32)
    plus_sens = np.asarray([row["plus_sensitivity"] for row in rows], dtype=np.float32)
    minus_sens = np.asarray([row["minus_sensitivity"] for row in rows], dtype=np.float32)
    return {
        "avg_original_output": float(np.mean(original)),
        "avg_plus_abs_change": float(np.mean(plus_change)),
        "avg_minus_abs_change": float(np.mean(minus_change)),
        "avg_abs_change": float(np.mean(np.concatenate([plus_change, minus_change]))),
        "avg_plus_sensitivity": float(np.mean(plus_sens)),
        "avg_minus_sensitivity": float(np.mean(minus_sens)),
        "avg_sensitivity": float(np.mean(np.concatenate([plus_sens, minus_sens]))),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    infer_cfg_path = Path(args.infer_config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = (root / infer_cfg_path).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = load_json(infer_cfg_path)
    X_scaled, y, source = build_dataset(root=root, infer_cfg=infer_cfg, data_dir=data_dir)
    if X_scaled.shape[0] == 0:
        raise RuntimeError(f"没有构建出任何样本: {data_dir}")

    model = build_float_model(root=root, infer_cfg=infer_cfg, input_shape=tuple(X_scaled.shape[1:]))
    feature_extractor = LSTMLastFeatureExtractor(model).eval()
    reg_head = model.fc.eval()

    high_candidates = np.flatnonzero(y >= np.float32(args.high_threshold)).astype(np.int64)
    low_candidates = np.flatnonzero(y < np.float32(args.high_threshold)).astype(np.int64)
    high_indices = evenly_pick_indices(high_candidates, int(args.samples_per_group))
    low_indices = evenly_pick_indices(low_candidates, int(args.samples_per_group))

    delta_value = float(args.delta)
    perturb_note = (
        f"对 Y[4] 的全部 50 维统一加减常数 {delta_value:.6f}，"
        "即对整个特征向量做同幅平移。"
    )

    rows = []
    with torch.no_grad():
        for group_name, indices in [("high_dryness", high_indices), ("non_high_dryness", low_indices)]:
            for sample_id, global_idx in enumerate(indices.tolist()):
                sample = X_scaled[global_idx : global_idx + 1].astype(np.float32)
                y4 = feature_extractor(torch.from_numpy(sample)).detach().cpu().numpy().astype(np.float32)
                base_output = reg_head(torch.from_numpy(y4)).detach().cpu().numpy().reshape(-1).astype(np.float32)[0]

                plus_input = y4 + np.float32(delta_value)
                minus_input = y4 - np.float32(delta_value)
                plus_output = reg_head(torch.from_numpy(plus_input)).detach().cpu().numpy().reshape(-1).astype(np.float32)[0]
                minus_output = reg_head(torch.from_numpy(minus_input)).detach().cpu().numpy().reshape(-1).astype(np.float32)[0]

                plus_abs_change = float(abs(plus_output - base_output))
                minus_abs_change = float(abs(minus_output - base_output))

                rows.append(
                    {
                        "group": group_name,
                        "global_index": int(global_idx),
                        "sample_id": int(sample_id),
                        "source_csv": str(source[global_idx]),
                        "true_label": float(y[global_idx]),
                        "original_output": float(base_output),
                        "plus_output": float(plus_output),
                        "minus_output": float(minus_output),
                        "plus_abs_change": plus_abs_change,
                        "minus_abs_change": minus_abs_change,
                        "plus_sensitivity": float(plus_abs_change / delta_value),
                        "minus_sensitivity": float(minus_abs_change / delta_value),
                        "y4_json": json.dumps(y4.reshape(-1).tolist(), ensure_ascii=False),
                    }
                )

    high_rows = [row for row in rows if row["group"] == "high_dryness"]
    low_rows = [row for row in rows if row["group"] == "non_high_dryness"]

    summary = {
        "experiment_setup": {
            "model_type": "PC float CNN-LSTM",
            "infer_config": str(infer_cfg_path),
            "data_dir": str(data_dir),
            "high_threshold": float(args.high_threshold),
            "samples_per_group": int(args.samples_per_group),
            "delta": delta_value,
            "perturbation": perturb_note,
            "selection_rule": "在高干度组和非高干度组内，分别按全组索引均匀抽取 5 个样本。",
        },
        "high_dryness_summary": summarize_group(high_rows),
        "non_high_dryness_summary": summarize_group(low_rows),
    }

    save_json(output_dir / "summary.json", summary)
    save_csv(output_dir / "details.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
