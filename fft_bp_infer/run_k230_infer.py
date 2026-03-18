import gc
import json
import math
import time
from pathlib import Path

try:
    import ulab.numpy as np  # type: ignore
except ImportError:
    import numpy as np  # type: ignore


def now_us():
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.perf_counter() * 1_000_000)


def diff_us(t_end, t_start):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(t_end, t_start)
    return t_end - t_start


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(field_name + " must be > 0, got " + str(parsed))
    return parsed


def resolve_positive_step(value, fallback: int, field_name: str) -> int:
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


def parse_label_from_name(filename: str) -> float:
    stem = Path(filename).stem
    if "-" not in stem:
        return float("nan")
    token = stem.split("-")[0]
    try:
        return float(token)
    except ValueError:
        return float("nan")


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
    n = 1
    while n < window_size:
        n <<= 1
    return n


def get_window(window_name: str, window_size: int):
    name = str(window_name).lower()
    if name in {"hann", "hanning"}:
        if hasattr(np, "hanning"):
            return np.hanning(window_size)
        return np.asarray(
            [0.5 - 0.5 * math.cos(2.0 * math.pi * i / max(1, window_size - 1)) for i in range(window_size)],
            dtype=np.float32,
        )
    if name == "hamming":
        if hasattr(np, "hamming"):
            return np.hamming(window_size)
        return np.asarray(
            [0.54 - 0.46 * math.cos(2.0 * math.pi * i / max(1, window_size - 1)) for i in range(window_size)],
            dtype=np.float32,
        )
    if name in {"rect", "boxcar", "none"}:
        return np.ones(window_size, dtype=np.float32)
    raise ValueError("Unsupported FFT window: " + str(window_name))


def fft_single_side_mag_fallback(x: np.ndarray, nfft: int):
    x = x.astype(np.float32)
    if x.shape[0] < nfft:
        pad = np.zeros((nfft - x.shape[0],), dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    elif x.shape[0] > nfft:
        x = x[:nfft]

    half = nfft // 2
    out = np.zeros((half + 1,), dtype=np.float32)
    for k in range(half + 1):
        re = 0.0
        im = 0.0
        for n in range(nfft):
            angle = -2.0 * math.pi * k * n / nfft
            v = float(x[n])
            re += v * math.cos(angle)
            im += v * math.sin(angle)
        mag = math.sqrt(re * re + im * im) / nfft
        if 0 < k < half:
            mag *= 2.0
        out[k] = mag
    return out


def fft_transform(data_1d: np.ndarray, fft_cfg: dict):
    x = np.asarray(data_1d, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x - float(np.mean(x))
    x = x * get_window(fft_cfg.get("window", "hann"), x.size)
    nfft = resolve_nfft(fft_cfg.get("nfft", "auto"), x.size)

    if hasattr(np, "fft") and hasattr(np.fft, "fft"):
        y = np.fft.fft(x, n=nfft)
        p2 = np.abs(y / nfft)
        p1 = p2[: nfft // 2 + 1]
        if p1.size > 2:
            p1[1:-1] = p1[1:-1] * 2.0
    else:
        p1 = fft_single_side_mag_fallback(x, nfft)

    fs = float(fft_cfg.get("fs", 1000.0))
    f_min = float(fft_cfg.get("f_min", 0.1))
    f_max = float(fft_cfg.get("f_max", 50.0))
    freqs = fs * np.arange(0, nfft // 2 + 1) / nfft
    mask = (freqs >= f_min) & (freqs < f_max)
    return p1[mask].astype(np.float32)


def build_dataset(cfg: dict, root: Path):
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    fft_cfg = cfg["preprocessing"]["fft_config"]
    data_dir = (root / paths["test_data_dir"]).resolve()

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    X_list = []
    y_list = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = read_signal(csv_file)
        if signal.size < base_window:
            continue
        features = []
        for start in range(0, signal.size - base_window + 1, base_step):
            feat = fft_transform(signal[start : start + base_window], fft_cfg)
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
    return np.stack(X_list).astype(np.float32), np.asarray(y_list, dtype=np.float32)


def scale_features(X: np.ndarray, scaler_json_path: Path):
    scaler = load_json(scaler_json_path)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    eps = 1e-12
    scale = np.where(np.abs(scale) < eps, 1.0, scale).astype(np.float32)
    X_flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    X_scaled = (X_flat - mean) / scale
    return X_scaled.reshape(X.shape).astype(np.float32)


def write_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("sample_id,true_label,prediction\n")
        for i in range(len(y_pred)):
            f.write("{},{},{}\n".format(i, float(y_true[i]), float(y_pred[i])))


def run_kmodel_inference(kmodel_path: Path, X_scaled: np.ndarray):
    import nncase_runtime as nn  # type: ignore

    kpu = nn.kpu()
    kpu.load_kmodel(str(kmodel_path))

    preds = []
    infer_us_total = 0
    for i in range(X_scaled.shape[0]):
        sample = X_scaled[i].astype(np.float32)
        sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
        input_tensor = nn.from_numpy(sample)
        kpu.set_input_tensor(0, input_tensor)
        t0 = now_us()
        kpu.run()
        t1 = now_us()
        infer_us_total += diff_us(t1, t0)
        output = kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        preds.append(pred)
        del output
        del input_tensor
        if (i + 1) % 64 == 0:
            gc.collect()
    return np.asarray(preds, dtype=np.float32), infer_us_total


def safe_metric_mae(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true)
    if hasattr(mask, "any") and bool(mask.any()):
        return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    return float("nan")


def safe_metric_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true)
    if hasattr(mask, "any") and bool(mask.any()):
        return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return float("nan")


def main():
    root = Path(__file__).resolve().parent
    cfg = load_json(root / "k230_config.json")
    paths = cfg["paths"]

    kmodel_path = (root / paths["kmodel"]).resolve()
    scaler_json_path = (root / paths["scaler_json"]).resolve()
    pred_csv = (root / paths["predictions_csv"]).resolve()

    t_start = now_us()
    X, y = build_dataset(cfg, root)
    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found in test_data.")
    X_scaled = scale_features(X, scaler_json_path)
    y_pred, infer_us = run_kmodel_inference(kmodel_path, X_scaled)
    t_end = now_us()

    write_predictions(pred_csv, y, y_pred)
    total_us = diff_us(t_end, t_start)
    mae = safe_metric_mae(y, y_pred)
    rmse = safe_metric_rmse(y, y_pred)

    print("=== K230 FFT+BP Inference ===")
    print("kmodel:", kmodel_path)
    print("samples:", int(X.shape[0]))
    print("input_shape:", tuple(X.shape[1:]))
    print("model_infer_time_sec:", infer_us / 1_000_000.0)
    print("model_infer_time_per_sample_ms:", infer_us / 1000.0 / float(X.shape[0]))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("prediction_csv:", pred_csv)
    print("first_10_predictions:", y_pred[:10].tolist())


if __name__ == "__main__":
    main()
