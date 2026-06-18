import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import build_kmodel


DEFAULT_SCHEMES = [
    {
        "id": "u8u8_kld512",
        "samples_count": 512,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "uint8",
        "weight_quant_type": "uint8",
        "calibrate_method": "Kld",
    },
    {
        "id": "i16u8_kld512",
        "samples_count": 512,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "int16",
        "weight_quant_type": "uint8",
        "calibrate_method": "Kld",
    },
    {
        "id": "u8i16_kld512",
        "samples_count": 512,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "uint8",
        "weight_quant_type": "int16",
        "calibrate_method": "Kld",
    },
    {
        "id": "u8u8_kld1024",
        "samples_count": 1024,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "uint8",
        "weight_quant_type": "uint8",
        "calibrate_method": "Kld",
    },
    {
        "id": "u8i16_kld1024",
        "samples_count": 1024,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "uint8",
        "weight_quant_type": "int16",
        "calibrate_method": "Kld",
    },
    {
        "id": "u8u8_noclip512",
        "samples_count": 512,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "uint8",
        "weight_quant_type": "uint8",
        "calibrate_method": "NoClip",
    },
    {
        "id": "i8i8_kld512",
        "samples_count": 512,
        "sampling_strategy": "per_dryness_uniform",
        "quant_type": "int8",
        "weight_quant_type": "int8",
        "calibrate_method": "Kld",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate seg3 quantization schemes with nncase Simulator.")
    parser.add_argument("--export_config", default="configs/export/k230_export_config_cnn_tcn_seg3.json")
    parser.add_argument("--output_dir", default="artifacts/seg3_quant_eval")
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--skip_build", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_scheme_cfg(base_cfg: dict, scheme: dict, output_dir: Path):
    cfg = deepcopy(base_cfg)
    cfg["name"] = "seg3_" + scheme["id"]
    cfg["quantization"].update(scheme)
    cfg["paths"]["onnx"] = str((output_dir / (scheme["id"] + ".onnx")).resolve())
    cfg["paths"]["kmodel"] = str((output_dir / (scheme["id"] + ".kmodel")).resolve())
    cfg["paths"]["scaler_json"] = str((output_dir / (scheme["id"] + "_scaler.json")).resolve())
    cfg["paths"]["calibration_npy"] = str((output_dir / (scheme["id"] + "_calibration.npy")).resolve())
    cfg["paths"]["nncase_dump_dir"] = str((output_dir / (scheme["id"] + "_nncase_dump")).resolve())
    return cfg


def build_dataset_and_pth(root: Path, cfg: dict, eval_samples: int):
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    data_dir = (root / paths["test_data_dir"]).resolve()
    feature_mode = build_kmodel.normalize_feature_mode(cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    X, y = build_kmodel.build_dataset(
        data_dir=data_dir,
        base_window_size=data_cfg["base_window_size"],
        base_step=data_cfg["base_step"],
        seq_length=data_cfg["sequence_length"],
        seq_step=data_cfg["sequence_step"],
        feature_mode=feature_mode,
    )
    if X.shape[0] == 0:
        raise RuntimeError("没有可用评估样本。")
    X_raw, y_raw = build_kmodel.build_seg3_raw_aux_dataset(
        data_dir=data_dir,
        base_window_size=data_cfg["base_window_size"],
        base_step=data_cfg["base_step"],
        seq_length=data_cfg["sequence_length"],
        seq_step=data_cfg["sequence_step"],
    )
    if X_raw.shape != X.shape or not np.array_equal(y_raw, y):
        raise RuntimeError("seg3 raw aux evaluation dataset is not aligned with main dataset.")
    count = min(int(eval_samples), int(X.shape[0]))
    indices = np.linspace(0, int(X.shape[0]) - 1, num=count, dtype=np.int64)
    X = X[indices].astype(np.float32)
    y = y[indices].astype(np.float32)
    X_raw = X_raw[indices].astype(np.float32)
    X_scaled = build_kmodel.apply_scaler((root / paths["scaler_pkl"]).resolve(), X)

    state = build_kmodel.load_state_dict_compat((root / paths["weights_pth"]).resolve(), torch.device("cpu"))
    model = build_kmodel.build_model_from_config(cfg["model"], tuple(X_scaled.shape[1:]), state)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        pth = model.compose_prediction(
            model(torch.from_numpy(X_scaled), aux=torch.from_numpy(X_raw))
        ).numpy().reshape(-1)
    return X_scaled, X_raw, y, pth.astype(np.float32)


def run_simulator(kmodel_path: Path, X_scaled: np.ndarray, X_raw: np.ndarray):
    build_kmodel.prepare_nncase_env()
    import nncase

    sim = nncase.Simulator()
    sim.load_model(kmodel_path.read_bytes())
    preds = []
    for i in range(int(X_scaled.shape[0])):
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(X_scaled[i : i + 1].astype(np.float32)))
        sim.set_input_tensor(1, nncase.RuntimeTensor.from_numpy(X_raw[i : i + 1].astype(np.float32)))
        sim.run()
        preds.append(float(sim.get_output_tensor(0).to_numpy().reshape(-1)[0]))
    return np.asarray(preds, dtype=np.float32)


def metrics(pred: np.ndarray, pth: np.ndarray, y: np.ndarray):
    diff = pred - pth
    truth_diff = pred - y
    return {
        "pth_vs_kmodel_mae": float(np.mean(np.abs(diff))),
        "pth_vs_kmodel_max_abs": float(np.max(np.abs(diff))),
        "kmodel_mae_vs_true": float(np.mean(np.abs(truth_diff))),
        "kmodel_min": float(np.min(pred)),
        "kmodel_max": float(np.max(pred)),
        "first5_kmodel": np.round(pred[:5], 6).tolist(),
        "first5_pth": np.round(pth[:5], 6).tolist(),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    export_config = (root / args.export_config).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = build_kmodel.load_json(export_config)

    schemes = DEFAULT_SCHEMES
    if args.scheme:
        schemes = [s for s in schemes if s["id"] == args.scheme]
        if not schemes:
            raise KeyError("Unknown scheme: " + args.scheme)

    X_scaled, X_raw, y, pth = build_dataset_and_pth(root, base_cfg, args.eval_samples)
    summary = []
    for scheme in schemes:
        print("=== scheme", scheme["id"], "===")
        cfg = make_scheme_cfg(base_cfg, scheme, output_dir)
        cfg_path = output_dir / (scheme["id"] + "_export_config.json")
        save_json(cfg_path, cfg)
        if not args.skip_build:
            build_kmodel.export_scaler_json((root / cfg["paths"]["scaler_pkl"]).resolve(), Path(cfg["paths"]["scaler_json"]))
            build_kmodel.main_for_external_config = None
            old_argv = None
            import sys

            old_argv = sys.argv[:]
            sys.argv = ["build_kmodel.py", "--config", str(cfg_path)]
            try:
                build_kmodel.main()
            finally:
                sys.argv = old_argv
        pred = run_simulator(Path(cfg["paths"]["kmodel"]), X_scaled, X_raw)
        item = {"scheme": scheme["id"], **scheme, **metrics(pred, pth, y)}
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    summary = sorted(summary, key=lambda x: x["pth_vs_kmodel_mae"])
    save_json(output_dir / "summary.json", summary)
    print("best_scheme:", summary[0]["scheme"])
    print("summary_path:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
