import gc
import json
import time

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore

try:
    import ulab.numpy as np  # type: ignore
except ImportError:
    import numpy as np  # type: ignore

NP_FLOAT = getattr(np, "float32", None)
if NP_FLOAT is None:
    NP_FLOAT = getattr(np, "float", None)
if NP_FLOAT is None:
    NP_FLOAT = float


def as_float_array(values):
    try:
        return np.asarray(values, dtype=NP_FLOAT)
    except TypeError:
        return np.asarray(values)


def astype_float_array(arr):
    if not hasattr(arr, "astype"):
        return arr
    try:
        return arr.astype(NP_FLOAT)
    except TypeError:
        return arr


def empty_float(shape):
    try:
        return np.empty(shape, dtype=NP_FLOAT)
    except TypeError:
        return np.empty(shape)


def now_us():
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.perf_counter() * 1_000_000)


def diff_us(t_end, t_start):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(t_end, t_start)
    return t_end - t_start


def norm_path(path):
    return str(path).replace("\\", "/")


def join_path(base, rel):
    rel = norm_path(rel)
    if rel.startswith("/"):
        return rel
    base = norm_path(base)
    if base.endswith("/"):
        return base + rel
    return base + "/" + rel


def dirname(path):
    p = norm_path(path).rstrip("/")
    idx = p.rfind("/")
    if idx < 0:
        return "."
    if idx == 0:
        return "/"
    return p[:idx]


def exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def ensure_dir(path):
    p = norm_path(path)
    if p in {"", ".", "/"}:
        return
    abs_path = p.startswith("/")
    cur = "/" if abs_path else ""
    parts = [seg for seg in p.strip("/").split("/") if seg]
    for seg in parts:
        if cur == "/":
            cur = "/" + seg
        elif cur == "":
            cur = seg
        else:
            cur = cur + "/" + seg
        try:
            os.mkdir(cur)
        except OSError:
            pass


def list_csv_files(data_dir):
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    files = []
    for name in names:
        if str(name).lower().endswith(".csv"):
            files.append(join_path(data_dir, name))
    files.sort()
    return files


def file_stem(path):
    name = norm_path(path).split("/")[-1]
    dot = name.rfind(".")
    if dot > 0:
        return name[:dot]
    return name


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def require_positive_int(value, field_name):
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(field_name + " must be > 0, got " + str(parsed))
    return parsed


def resolve_positive_step(value, fallback, field_name):
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


def parse_label_from_name(filename):
    stem = file_stem(filename)
    if "-" not in stem:
        return float("nan")
    token = stem.split("-")[0]
    try:
        return float(token)
    except ValueError:
        return float("nan")


def read_signal(csv_path):
    values = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",")[0].strip()
            try:
                values.append(float(first))
            except ValueError:
                continue
    return as_float_array(values)


def build_dataset(cfg, root):
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    data_dir = join_path(root, paths["test_data_dir"])

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    X_list = []
    y_list = []
    for csv_file in list_csv_files(data_dir):
        signal = read_signal(csv_file)
        if signal.size < base_window:
            continue
        features = []
        for start in range(0, signal.size - base_window + 1, base_step):
            features.append(astype_float_array(signal[start : start + base_window]))
        if len(features) < seq_length:
            continue
        label = parse_label_from_name(csv_file)
        for i in range(0, len(features) - seq_length + 1, seq_step):
            sample = empty_float((seq_length, base_window))
            seg = features[i : i + seq_length]
            for j in range(seq_length):
                sample[j] = seg[j]
            X_list.append(sample)
            y_list.append(label)

    if not X_list:
        return empty_float((0, seq_length, 0)), empty_float((0,))

    sample_width = int(X_list[0].shape[1]) if len(X_list) > 0 else 0
    X = empty_float((len(X_list), seq_length, sample_width))
    for i in range(len(X_list)):
        X[i] = X_list[i]
    return astype_float_array(X), as_float_array(y_list)


def scale_features(X, scaler_json_path):
    scaler = load_json(scaler_json_path)
    mean = as_float_array(scaler["mean"])
    scale = as_float_array(scaler["scale"])
    eps = 1e-12
    for i in range(len(scale)):
        if abs(float(scale[i])) < eps:
            scale[i] = 1.0
    X_flat = astype_float_array(X.reshape((X.shape[0] * X.shape[1], X.shape[-1])))
    X_scaled = (X_flat - mean) / scale
    return astype_float_array(X_scaled.reshape(X.shape))


def write_predictions(path, y_true, y_pred):
    ensure_dir(dirname(path))
    with open(path, "w") as f:
        f.write("sample_id,true_label,prediction\n")
        for i in range(len(y_pred)):
            f.write("{},{},{}\n".format(i, float(y_true[i]), float(y_pred[i])))


def run_kmodel_inference(kmodel_path, X_scaled):
    import nncase_runtime as nn  # type: ignore

    kpu = nn.kpu()
    kpu.load_kmodel(kmodel_path)

    preds = []
    infer_us_total = 0
    for i in range(X_scaled.shape[0]):
        sample = astype_float_array(X_scaled[i])
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
    return as_float_array(preds), infer_us_total


def safe_metric_mae(y_true, y_pred):
    total = 0.0
    count = 0
    for i in range(len(y_pred)):
        t = float(y_true[i])
        p = float(y_pred[i])
        if t == t:
            d = p - t
            if d < 0:
                d = -d
            total += d
            count += 1
    if count == 0:
        return float("nan")
    return total / float(count)


def safe_metric_rmse(y_true, y_pred):
    total = 0.0
    count = 0
    for i in range(len(y_pred)):
        t = float(y_true[i])
        p = float(y_pred[i])
        if t == t:
            d = p - t
            total += d * d
            count += 1
    if count == 0:
        return float("nan")
    return float(np.sqrt(total / float(count)))


def detect_root():
    candidates = []
    try:
        candidates.append(norm_path(os.getcwd()))
    except Exception:
        pass
    here = globals().get("__file__", "")
    if here:
        here = norm_path(here)
        if "/" in here:
            candidates.append(dirname(here))
    candidates.append("/sdcard/raw_cnn_infer")
    candidates.append("/sdcard")

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        if exists(join_path(c, "k230_config.json")):
            return c
    return ordered[0] if ordered else "."


def main():
    root = detect_root()
    cfg = load_json(join_path(root, "k230_config.json"))
    paths = cfg["paths"]

    kmodel_path = join_path(root, paths["kmodel"])
    scaler_json_path = join_path(root, paths["scaler_json"])
    pred_csv = join_path(root, paths["predictions_csv"])

    t_start = now_us()
    X, y = build_dataset(cfg, root)
    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found in test_data.")
    # Quick smoke-test mode: only run the first 10 samples.
    if X.shape[0] > 10:
        X = X[:10]
        y = y[:10]
    X_scaled = scale_features(X, scaler_json_path)
    y_pred, infer_us = run_kmodel_inference(kmodel_path, X_scaled)
    t_end = now_us()

    write_predictions(pred_csv, y, y_pred)
    total_us = diff_us(t_end, t_start)
    mae = safe_metric_mae(y, y_pred)
    rmse = safe_metric_rmse(y, y_pred)

    print("=== K230 Raw+CNN Inference ===")
    print("root:", root)
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
