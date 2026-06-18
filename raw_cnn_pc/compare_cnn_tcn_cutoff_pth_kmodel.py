import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

import build_kmodel
import compare_pth_kmodel


STAGES = [
    "window_conv0_relu",
    "window_pool0",
    "window_conv1_relu",
    "window_pool1",
    "window_mean",
    "tcn_input",
    "tcn_block0",
    "tcn_block1",
    "tcn_block2",
    "tcn_mean",
    "head_output",
]


class CNNTCNCutoff(nn.Module):
    def __init__(self, model: nn.Module, stage: str):
        super().__init__()
        self.model = model
        self.stage = stage

    def forward(self, x):
        batch_size, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * time_steps, 1, feature_dim)

        x = torch.relu(self.model.window_convs[0](x))
        if self.stage == "window_conv0_relu":
            return x
        x = self.model.window_pools[0](x)
        if self.stage == "window_pool0":
            return x

        x = torch.relu(self.model.window_convs[1](x))
        if self.stage == "window_conv1_relu":
            return x
        x = self.model.window_pools[1](x)
        if self.stage == "window_pool1":
            return x

        x = x.mean(dim=-1)
        x = x.reshape(batch_size, time_steps, -1)
        if self.stage == "window_mean":
            return x

        x = x.transpose(1, 2)
        if self.stage == "tcn_input":
            return x

        for idx, block in enumerate(self.model.temporal_network):
            x = block(x)
            if self.stage == "tcn_block{}".format(idx):
                return x

        x = torch.mean(x, dim=2)
        if self.stage == "tcn_mean":
            return x

        return self.model.head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CNN-TCN cutoff PTH outputs against cutoff kmodels.")
    parser.add_argument("--infer_config", type=str, required=True)
    parser.add_argument("--export_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/layerwise/cnn_tcn_cutoff")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--csv_name", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_dryness", type=int, default=None)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--stages", type=str, default=",".join(STAGES))
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "shape",
                "pth_min",
                "pth_max",
                "kmodel_min",
                "kmodel_max",
                "mae",
                "rmse",
                "max_abs",
                "relative_mae",
            ]
        )
        writer.writerows(rows)


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def build_scaled_dataset(root: Path, infer_cfg: dict, data_dir: Path, max_samples=None, max_per_dryness=None):
    data_cfg = infer_cfg["data"]
    feature_mode = build_kmodel.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    X, y, source = compare_pth_kmodel.build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=compare_pth_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=compare_pth_kmodel.require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=compare_pth_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=compare_pth_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=feature_mode,
        max_samples=max_samples,
    )
    # 和四路对比报告保持同一套采样限制，确保 sample_index 能对齐。
    X, y, source = compare_pth_kmodel.apply_sample_limits(
        X,
        y,
        source,
        max_samples=max_samples,
        max_per_dryness=max_per_dryness,
    )
    scaler_path = resolve_path(root, infer_cfg["normalization"]["scaler_path"])
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    return X_scaled, y.astype(np.float32), source


def choose_sample(X_scaled, y, source, sample_index, csv_name):
    if csv_name:
        mask = source == csv_name
        if not np.any(mask):
            raise FileNotFoundError("csv_name not found in samples: {}".format(csv_name))
        X_scaled = X_scaled[mask]
        y = y[mask]
        source = source[mask]
    if sample_index < 0 or sample_index >= int(X_scaled.shape[0]):
        raise IndexError("sample_index out of range: {}, total={}".format(sample_index, X_scaled.shape[0]))
    return X_scaled[sample_index : sample_index + 1], float(y[sample_index]), str(source[sample_index])


def build_base_model(root: Path, infer_cfg: dict, input_shape):
    model_cfg = infer_cfg["model"]
    weights_path = resolve_path(root, model_cfg["weights_path"])
    state_dict = build_kmodel.load_state_dict_compat(weights_path, torch.device("cpu"))
    model = build_kmodel.build_model_from_config(model_cfg=model_cfg, input_shape=input_shape, state_dict=state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_cutoff_onnx(model, sample_shape, onnx_path: Path):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, *sample_shape, dtype=torch.float32)
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
    import onnx

    build_kmodel.sanitize_onnx_for_nncase(onnx_path, onnx)


def clone_export_cfg(export_cfg: dict, output_dir: Path, stage: str):
    cfg = json.loads(json.dumps(export_cfg))
    cfg["name"] = cfg.get("name", "cnn_tcn") + "_cutoff_" + stage
    paths = cfg["paths"]
    paths["onnx"] = (output_dir / "{}.onnx".format(stage)).resolve().as_posix()
    paths["kmodel"] = (output_dir / "{}.kmodel".format(stage)).resolve().as_posix()
    paths["nncase_dump_dir"] = (output_dir / "{}_nncase_dump".format(stage)).resolve().as_posix()
    return cfg


def run_kmodel(sample: np.ndarray, kmodel_path: Path):
    import nncase

    sim = nncase.Simulator()
    sim.load_model(kmodel_path.read_bytes())
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(sample.astype(np.float32)))
    sim.run()
    return sim.get_output_tensor(0).to_numpy().astype(np.float32)


def metrics(stage, pth_value, kmodel_value):
    p = np.asarray(pth_value, dtype=np.float32)
    k = np.asarray(kmodel_value, dtype=np.float32)
    diff = k - p
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.mean(np.abs(p))) + 1e-12
    return [
        stage,
        list(p.shape),
        float(np.min(p)),
        float(np.max(p)),
        float(np.min(k)),
        float(np.max(k)),
        mae,
        rmse,
        max_abs,
        float(mae / denom),
    ]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    infer_cfg = load_json(resolve_path(root, args.infer_config))
    export_cfg = load_json(resolve_path(root, args.export_config))
    data_dir = resolve_path(root, args.data_dir)
    output_dir = resolve_path(root, args.output_dir)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage in stages:
        if stage not in STAGES:
            raise ValueError("unsupported stage: {}".format(stage))

    X_scaled, y, source = build_scaled_dataset(
        root,
        infer_cfg,
        data_dir,
        max_samples=args.max_samples,
        max_per_dryness=args.max_per_dryness,
    )
    sample, label, csv_name = choose_sample(X_scaled, y, source, args.sample_index, args.csv_name)
    input_shape = tuple(sample.shape[1:])
    base_model = build_base_model(root, infer_cfg, input_shape)

    qcfg = export_cfg.get("quantization", {})
    if args.max_calib_samples is not None:
        calib_count = build_kmodel.resolve_calibration_sample_count(args.max_calib_samples, X_scaled.shape[0], "max_calib_samples")
    else:
        calib_count = build_kmodel.resolve_calibration_sample_count(qcfg.get("samples_count", 64), X_scaled.shape[0], "quantization.samples_count")
    calibration_data = build_kmodel.select_calibration_data(
        X_scaled,
        count=calib_count,
        strategy=qcfg.get("sampling_strategy", "first"),
        random_seed=qcfg.get("random_seed", None),
        y_labels=y,
    )

    rows = []
    for stage in stages:
        cutoff = CNNTCNCutoff(base_model, stage).eval()
        stage_dir = output_dir / stage
        stage_cfg = clone_export_cfg(export_cfg, stage_dir, stage)
        onnx_path = Path(stage_cfg["paths"]["onnx"])
        kmodel_path = Path(stage_cfg["paths"]["kmodel"])
        export_cutoff_onnx(cutoff, input_shape, onnx_path)
        build_kmodel.compile_kmodel_with_nncase(stage_cfg, root, calibration_data)

        with torch.no_grad():
            pth_value = cutoff(torch.from_numpy(sample.astype(np.float32))).detach().cpu().numpy().astype(np.float32)
        kmodel_value = run_kmodel(sample, kmodel_path)
        rows.append(metrics(stage, pth_value, kmodel_value))
        print("stage={}, mae={:.8f}, max_abs={:.8f}".format(stage, rows[-1][6], rows[-1][8]))

    save_csv(output_dir / "cutoff_metrics.csv", rows)
    save_json(
        output_dir / "cutoff_summary.json",
        {
            "data_dir": str(data_dir),
            "sample_index": int(args.sample_index),
            "max_samples": args.max_samples,
            "max_per_dryness": args.max_per_dryness,
            "csv_name": csv_name,
            "label": label,
            "input_shape": list(input_shape),
            "calibration_samples": int(calibration_data.shape[0]),
            "metrics_csv": str((output_dir / "cutoff_metrics.csv").resolve()),
        },
    )
    print("metrics_csv:", (output_dir / "cutoff_metrics.csv").resolve())
    print("summary_json:", (output_dir / "cutoff_summary.json").resolve())


if __name__ == "__main__":
    main()
