import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Build K230 deploy assets for FFT+BP.")
    parser.add_argument("--config", type=str, default="k230_config.json")
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


def resolve_nfft(nfft_cfg, window_size: int):
    if isinstance(nfft_cfg, int) and nfft_cfg > 0:
        return int(nfft_cfg)
    return 1 << (window_size - 1).bit_length()


def get_window(window_name: str, window_size: int):
    name = str(window_name).lower()
    if name in {"hann", "hanning"}:
        if hasattr(np, "hanning"):
            return np.hanning(window_size)
        # fallback
        return np.asarray(
            [0.5 - 0.5 * math.cos(2.0 * math.pi * i / max(1, window_size - 1)) for i in range(window_size)],
            dtype=np.float64,
        )
    if name == "hamming":
        if hasattr(np, "hamming"):
            return np.hamming(window_size)
        return np.asarray(
            [0.54 - 0.46 * math.cos(2.0 * math.pi * i / max(1, window_size - 1)) for i in range(window_size)],
            dtype=np.float64,
        )
    if name in {"rect", "boxcar", "none"}:
        return np.ones(window_size, dtype=np.float64)
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
    def __init__(self, input_shape, hidden_units, activation="tanh", output_activation="none"):
        super().__init__()
        time_steps, features = input_shape
        input_dim = int(time_steps) * int(features)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if not hidden_units:
            raise ValueError("hidden_units must not be empty")
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


def export_onnx(model: nn.Module, onnx_path: Path, input_shape):
    try:
        import onnx  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires `onnx` package. Install requirements_k230_host.txt first."
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, *input_shape, dtype=torch.float32)
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


def export_scaler_json(scaler_pkl: Path, scaler_json: Path):
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


def compile_kmodel_with_nncase(cfg: dict, root: Path, calibration_data: np.ndarray):
    try:
        import nncase  # type: ignore
    except ImportError as exc:
        raise RuntimeError("nncase is not installed in current environment.") from exc

    paths = cfg["paths"]
    qcfg = cfg["quantization"]
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
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = dump_dir.as_posix()

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = int(calibration_data.shape[0])
    ptq_options.quant_type = qcfg.get("quant_type", "uint8")
    ptq_options.w_quant_type = qcfg.get("weight_quant_type", "uint8")
    ptq_options.calibrate_method = qcfg.get("calibrate_method", "NoClip")

    sample_list = [calibration_data[i : i + 1].astype(np.float32) for i in range(calibration_data.shape[0])]
    ptq_options.set_tensor_data([sample_list])

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(model_content, import_options)
    compiler.use_ptq(ptq_options)
    compiler.compile()
    kmodel = compiler.gencode_tobytes()

    with kmodel_path.open("wb") as f:
        f.write(kmodel)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_json(cfg_path)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]
    qcfg = cfg["quantization"]
    model_cfg = cfg["model"]

    weights_pth = (root / paths["weights_pth"]).resolve()
    onnx_path = (root / paths["onnx"]).resolve()
    scaler_pkl = (root / paths["scaler_pkl"]).resolve()
    scaler_json = (root / paths["scaler_json"]).resolve()
    calib_npy = (root / paths["calibration_npy"]).resolve()
    test_data_dir = (root / paths["test_data_dir"]).resolve()

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    try:
        X, y = build_dataset(
            data_dir=test_data_dir,
            base_window_size=base_window,
            base_step=base_step,
            seq_length=seq_length,
            seq_step=seq_step,
            fft_cfg=pre_cfg["fft_config"],
        )
        if X.shape[0] == 0:
            raise RuntimeError(f"No valid samples in test data: {test_data_dir}")

        X_scaled = apply_scaler(scaler_pkl, X)
        if args.max_calib_samples is not None:
            requested = require_positive_int(args.max_calib_samples, "max_calib_samples")
        else:
            requested = require_positive_int(qcfg.get("samples_count", 64), "quantization.samples_count")
        count = min(requested, X_scaled.shape[0])
        calibration_data = X_scaled[:count].astype(np.float32)
        calib_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(calib_npy, calibration_data)

        export_scaler_json(scaler_pkl, scaler_json)

        input_shape = tuple(X_scaled.shape[1:])
        model = BPNet(
            input_shape=input_shape,
            hidden_units=model_cfg["hidden_units"],
            activation=model_cfg["activation"],
            output_activation=model_cfg.get("output_activation", "none"),
        )
        state_dict = normalize_bp_state_dict_keys(load_state_dict_compat(weights_pth, torch.device("cpu")))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        export_onnx(model, onnx_path, input_shape=input_shape)

        print("Exported ONNX:", onnx_path)
        print("Exported scaler json:", scaler_json)
        print("Saved calibration data:", calib_npy, calibration_data.shape)

        if args.skip_compile:
            print("Skip nncase compile (--skip_compile set).")
            return

        compile_kmodel_with_nncase(cfg, root, calibration_data)
        print("Generated kmodel:", (root / paths["kmodel"]).resolve())
    except RuntimeError as exc:
        print("ERROR:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
