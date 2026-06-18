import csv
import json
import os
import site
import threading
import time
import traceback
from pathlib import Path
from tkinter import END, Button, Entry, Label, Radiobutton, StringVar, Text, Tk, filedialog, messagebox

import joblib
import numpy as np
import onnxruntime as ort
import torch

import infer


DEFAULT_INFER_CONFIG = "configs/infer/infer_config_cnn_tcn.json"
DEFAULT_OUTPUT_DIR = "artifacts/pth_onnx_kmodel_compare"


def _bootstrap_nncase_env():
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

    seen = set()
    for raw in site_roots:
        if not raw:
            continue
        root = Path(raw).resolve()
        root_text = str(root)
        if root_text in seen:
            continue
        seen.add(root_text)

        candidate = root / "nncase" / "modules" / "kpu"
        if candidate.exists() and not os.environ.get("NNCASE_PLUGIN_PATH"):
            os.environ["NNCASE_PLUGIN_PATH"] = candidate.as_posix()

        current_path = os.environ.get("PATH", "")
        path_items = current_path.split(os.pathsep) if current_path else []
        if root_text not in path_items:
            os.environ["PATH"] = root_text + os.pathsep + current_path if current_path else root_text


_bootstrap_nncase_env()

import nncase


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_rows_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def resolve_under_root(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def model_tag(*paths: Path) -> str:
    parts = [path.stem if path.is_file() else path.name for path in paths if path]
    raw = "__".join(parts)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def find_one_file(model_dir: Path, patterns):
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("{} must be > 0, got {}".format(field_name, parsed))
    return parsed


def apply_pth_dir_overrides(infer_cfg: dict, model_dir: Path):
    # 让 pth 和 scaler 跟着模型目录走，避免每次手工改配置里的两处路径。
    pth_path = find_one_file(model_dir, ["*.pth", "**/*.pth"])
    scaler_path = find_one_file(model_dir, ["scaler*.pkl", "**/scaler*.pkl", "*.pkl", "**/*.pkl"])
    if pth_path is None:
        raise FileNotFoundError("pth 妯″瀷鐩綍閲屾病鏈夋壘鍒?.pth: {}".format(model_dir))
    if scaler_path is None:
        raise FileNotFoundError("pth 妯″瀷鐩綍閲屾病鏈夋壘鍒?scaler .pkl: {}".format(model_dir))
    cfg = json.loads(json.dumps(infer_cfg))
    cfg["model"]["weights_path"] = str(pth_path)
    cfg["normalization"]["scaler_path"] = str(scaler_path)
    return cfg, pth_path, scaler_path


def build_dataset_with_sources(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str,
    max_samples,
):
    X_list = []
    y_list = []
    source_list = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = infer.read_signal(csv_file)
        if signal.size < base_window_size:
            continue

        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            window = infer.apply_feature_mode(window.astype(np.float32), feature_mode)
            features.append(window)

        if len(features) < seq_length:
            continue

        label = infer.parse_label_from_name(csv_file.name)
        for i in range(0, len(features) - seq_length + 1, seq_step):
            X_list.append(np.stack(features[i : i + seq_length], axis=0))
            y_list.append(label)
            source_list.append(csv_file.name)
            if max_samples is not None and len(X_list) >= max_samples:
                return (
                    np.stack(X_list).astype(np.float32),
                    np.asarray(y_list, dtype=np.float32),
                    np.asarray(source_list, dtype=object),
                )

    if not X_list:
        return (
            np.empty((0, seq_length, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(source_list, dtype=object),
    )


def build_seg3_raw_aux_with_sources(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    max_samples,
):
    """中文注释：为 seg3 构造未做特征变换的原始窗口序列，并保留来源文件顺序。"""
    return build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=base_window_size,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode="raw",
        max_samples=max_samples,
    )


def build_scaled_dataset(root: Path, infer_cfg: dict, data_dir: Path):
    data_cfg = infer_cfg["data"]
    feature_mode = infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    X, y, source = build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=feature_mode,
        max_samples=None,
    )
    if X.shape[0] == 0:
        raise RuntimeError("娌℃湁鏈夋晥鏍锋湰: {}".format(data_dir))
    scaler_path = resolve_under_root(root, infer_cfg["normalization"]["scaler_path"])
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    sample_ids = np.arange(X_scaled.shape[0], dtype=np.int64)
    X_raw_aux = None
    if infer.normalize_model_type(infer_cfg["model"].get("type", "CNN-All")) == "cnn_tcn_seg3_soft_stats_moe":
        X_raw_aux, y_raw_aux, source_raw = build_seg3_raw_aux_with_sources(
            data_dir=data_dir,
            base_window_size=require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
            base_step=require_positive_int(data_cfg["base_step"], "data.base_window_size"),
            seq_length=require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
            seq_step=require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
            max_samples=None,
        )
        if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y) or not np.array_equal(source_raw, source):
            raise RuntimeError("seg3 raw aux dataset is not aligned with main dataset.")
        X_raw_aux = X_raw_aux.astype(np.float32)
    return X_scaled, y.astype(np.float32), source, sample_ids, feature_mode, X_raw_aux


def run_pth_predictions(X_scaled: np.ndarray, infer_cfg: dict, root: Path, X_raw_aux=None):
    model_cfg = infer_cfg["model"]
    model_path = resolve_under_root(root, model_cfg["weights_path"])
    state_dict = infer.load_state_dict_compat(model_path, torch.device("cpu"))
    model = infer.build_model_from_config(
        model_cfg=model_cfg,
        input_shape=tuple(X_scaled.shape[1:]),
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        if infer.normalize_model_type(model_cfg.get("type", "CNN-All")) == "cnn_tcn_seg3_soft_stats_moe":
            if X_raw_aux is None:
                raise RuntimeError("seg3 requires raw aux input for PTH prediction.")
            outputs = model(
                torch.from_numpy(X_scaled),
                aux=torch.from_numpy(X_raw_aux.astype(np.float32, copy=False)),
            )
            return model.compose_prediction(outputs["prediction"]).cpu().numpy().reshape(-1).astype(np.float32)
        return model(torch.from_numpy(X_scaled)).cpu().numpy().reshape(-1).astype(np.float32)


def run_kmodel_predictions(X_scaled: np.ndarray, kmodel_path: Path, log_every: int, X_raw_aux=None):
    sim = nncase.Simulator()
    sim.load_model(kmodel_path.read_bytes())
    preds = np.empty((X_scaled.shape[0],), dtype=np.float32)
    total = int(X_scaled.shape[0])
    t0 = time.perf_counter()
    for i in range(total):
        sample = X_scaled[i : i + 1].astype(np.float32)
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(sample))
        if X_raw_aux is not None:
            raw_sample = X_raw_aux[i : i + 1].astype(np.float32)
            sim.set_input_tensor(1, nncase.RuntimeTensor.from_numpy(raw_sample))
        sim.run()
        preds[i] = float(np.array(sim.get_output_tensor(0).to_numpy()).reshape(-1)[0])
        if log_every > 0 and ((i + 1) % int(log_every) == 0 or (i + 1) == total):
            elapsed = time.perf_counter() - t0
            print(
                "kmodel_progress: {}/{} ({:.2f}%), elapsed_sec={:.3f}".format(
                    i + 1,
                    total,
                    (i + 1) * 100.0 / float(total),
                    elapsed,
                )
            )
    return preds


def run_onnx_predictions(X_scaled: np.ndarray, onnx_path: Path, X_raw_aux=None) -> np.ndarray:
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_name = session.get_outputs()[0].name
    preds = np.empty((X_scaled.shape[0],), dtype=np.float32)
    for idx in range(X_scaled.shape[0]):
        feed = {input_names[0]: X_scaled[idx : idx + 1].astype(np.float32)}
        if len(input_names) > 1:
            if X_raw_aux is None:
                raise RuntimeError("seg3 ONNX requires raw aux input.")
            feed[input_names[1]] = X_raw_aux[idx : idx + 1].astype(np.float32)
        pred = session.run([output_name], feed)[0]
        preds[idx] = float(np.asarray(pred, dtype=np.float32).reshape(-1)[0])
    return preds


def select_samples(X, y, source, sample_ids, mode: str, value: str, seed: str):
    if mode == "all":
        return X, y, source, sample_ids, {"compare_mode": "all", "compare_value": None}

    count = int(value)
    if count <= 0:
        raise ValueError("鏍锋湰鏁伴噺蹇呴』 > 0")

    if mode == "first_n":
        keep = np.arange(min(count, len(y)), dtype=np.int64)
        return X[keep], y[keep], source[keep], sample_ids[keep], {"compare_mode": "first_n", "compare_value": count}

    if mode == "random_per_dryness":
        rng = np.random.default_rng(int(seed))
        keep_parts = []
        for dryness in sorted({float(v) for v in y.tolist()}):
            indices = np.where(y == np.float32(dryness))[0]
            if len(indices) > count:
                indices = rng.choice(indices, size=count, replace=False)
            keep_parts.extend(indices.tolist())
        keep = np.asarray(sorted(keep_parts), dtype=np.int64)
        return (
            X[keep],
            y[keep],
            source[keep],
            sample_ids[keep],
            {"compare_mode": "random_per_dryness", "compare_value": count, "random_seed": int(seed)},
        )

    raise ValueError("鏈煡瀵规瘮妯″紡: {}".format(mode))


def make_summary(y_true, pth_pred, onnx_pred, kmodel_pred):
    pth_err = np.abs(pth_pred - y_true)
    onnx_err = np.abs(onnx_pred - y_true)
    kmodel_err = np.abs(kmodel_pred - y_true)
    pth_onnx = np.abs(pth_pred - onnx_pred)
    pth_kmodel = np.abs(pth_pred - kmodel_pred)
    onnx_kmodel = np.abs(onnx_pred - kmodel_pred)
    return {
        "total_samples": int(len(y_true)),
        "pth_mae_vs_true": float(np.mean(pth_err)),
        "onnx_mae_vs_true": float(np.mean(onnx_err)),
        "kmodel_mae_vs_true": float(np.mean(kmodel_err)),
        "pth_rmse_vs_true": float(np.sqrt(np.mean((pth_pred - y_true) ** 2))),
        "onnx_rmse_vs_true": float(np.sqrt(np.mean((onnx_pred - y_true) ** 2))),
        "kmodel_rmse_vs_true": float(np.sqrt(np.mean((kmodel_pred - y_true) ** 2))),
        "pth_vs_onnx_mae": float(np.mean(pth_onnx)),
        "pth_vs_kmodel_mae": float(np.mean(pth_kmodel)),
        "onnx_vs_kmodel_mae": float(np.mean(onnx_kmodel)),
        "pth_vs_onnx_rmse": float(np.sqrt(np.mean((pth_pred - onnx_pred) ** 2))),
        "pth_vs_kmodel_rmse": float(np.sqrt(np.mean((pth_pred - kmodel_pred) ** 2))),
        "onnx_vs_kmodel_rmse": float(np.sqrt(np.mean((onnx_pred - kmodel_pred) ** 2))),
        "pth_vs_onnx_max_abs": float(np.max(pth_onnx)),
        "pth_vs_kmodel_max_abs": float(np.max(pth_kmodel)),
        "onnx_vs_kmodel_max_abs": float(np.max(onnx_kmodel)),
    }


def make_detail_rows(sample_ids, source, y_true, pth_pred, onnx_pred, kmodel_pred):
    rows = []
    for idx in range(len(y_true)):
        rows.append(
            [
                int(sample_ids[idx]),
                str(source[idx]),
                float(y_true[idx]),
                float(pth_pred[idx]),
                float(onnx_pred[idx]),
                float(kmodel_pred[idx]),
                float(abs(pth_pred[idx] - y_true[idx])),
                float(abs(onnx_pred[idx] - y_true[idx])),
                float(abs(kmodel_pred[idx] - y_true[idx])),
                float(abs(pth_pred[idx] - onnx_pred[idx])),
                float(abs(pth_pred[idx] - kmodel_pred[idx])),
                float(abs(onnx_pred[idx] - kmodel_pred[idx])),
            ]
        )
    rows.sort(key=lambda row: row[11], reverse=True)
    return rows


def make_per_csv_rows(source, y_true, pth_pred, onnx_pred, kmodel_pred):
    rows = []
    for name in list(dict.fromkeys(source.tolist())):
        mask = source == name
        rows.append(
            [
                name,
                float(y_true[mask][0]),
                int(np.sum(mask)),
                float(np.mean(np.abs(pth_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(onnx_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(kmodel_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(pth_pred[mask] - onnx_pred[mask]))),
                float(np.mean(np.abs(pth_pred[mask] - kmodel_pred[mask]))),
                float(np.mean(np.abs(onnx_pred[mask] - kmodel_pred[mask]))),
                float(np.max(np.abs(onnx_pred[mask] - kmodel_pred[mask]))),
            ]
        )
    rows.sort(key=lambda row: row[8], reverse=True)
    return rows


def make_per_dryness_rows(y_true, pth_pred, onnx_pred, kmodel_pred):
    rows = []
    for dryness in sorted({float(v) for v in y_true.tolist()}):
        mask = y_true == np.float32(dryness)
        rows.append(
            [
                float(dryness),
                int(np.sum(mask)),
                float(np.mean(np.abs(pth_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(onnx_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(kmodel_pred[mask] - y_true[mask]))),
                float(np.mean(np.abs(pth_pred[mask] - onnx_pred[mask]))),
                float(np.mean(np.abs(pth_pred[mask] - kmodel_pred[mask]))),
                float(np.mean(np.abs(onnx_pred[mask] - kmodel_pred[mask]))),
                float(np.max(np.abs(onnx_pred[mask] - kmodel_pred[mask]))),
            ]
        )
    rows.sort(key=lambda row: row[7], reverse=True)
    return rows


def run_compare(infer_config: Path, pth_dir: Path, onnx_path: Path, kmodel_path: Path, data_dir: Path, output_dir: Path, mode: str, value: str, seed: str, log):
    root = Path(__file__).resolve().parent
    infer_cfg = load_json(infer_config)
    infer_cfg, pth_path, scaler_path = apply_pth_dir_overrides(infer_cfg, pth_dir)
    if not onnx_path.exists():
        raise FileNotFoundError("onnx 涓嶅瓨鍦? {}".format(onnx_path))
    if not kmodel_path.exists():
        raise FileNotFoundError("kmodel 涓嶅瓨鍦? {}".format(kmodel_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = model_tag(pth_path, onnx_path, kmodel_path)
    details_csv = output_dir / "compare_pth_onnx_kmodel_details__{}.csv".format(tag)
    per_csv_csv = output_dir / "compare_pth_onnx_kmodel_per_csv__{}.csv".format(tag)
    per_dryness_csv = output_dir / "compare_pth_onnx_kmodel_per_dryness__{}.csv".format(tag)
    summary_json = output_dir / "compare_pth_onnx_kmodel_summary__{}.json".format(tag)

    t0 = time.perf_counter()
    log("璇诲彇鏁版嵁骞舵爣鍑嗗寲: {}".format(data_dir))
    X_scaled, y_true, source, sample_ids, feature_mode, X_raw_aux = build_scaled_dataset(root, infer_cfg, data_dir)
    raw_total = int(X_scaled.shape[0])
    y_true_all = y_true
    source_all = source
    sample_ids_all = sample_ids
    X_scaled, y_true, source, sample_ids, selection = select_samples(X_scaled, y_true, source, sample_ids, mode, value, seed)
    if X_raw_aux is not None:
        X_raw_aux, y_aux, source_aux, sample_ids_aux, _ = select_samples(
            X_raw_aux,
            y_true_all,
            source_all,
            sample_ids_all,
            mode,
            value,
            seed,
        )
        if not np.array_equal(y_aux, y_true) or not np.array_equal(source_aux, source) or not np.array_equal(sample_ids_aux, sample_ids):
            raise RuntimeError("seg3 raw aux dataset is not aligned after sample selection.")
    log("鍙備笌瀵规瘮鏍锋湰鏁? {} / {}".format(X_scaled.shape[0], raw_total))

    t_pth = time.perf_counter()
    pth_pred = run_pth_predictions(X_scaled, infer_cfg, root, X_raw_aux=X_raw_aux)
    pth_sec = time.perf_counter() - t_pth

    t_onnx = time.perf_counter()
    onnx_pred = run_onnx_predictions(X_scaled, onnx_path, X_raw_aux=X_raw_aux)
    onnx_sec = time.perf_counter() - t_onnx

    t_kmodel = time.perf_counter()
    kmodel_pred = run_kmodel_predictions(X_scaled, kmodel_path, log_every=500, X_raw_aux=X_raw_aux)
    kmodel_sec = time.perf_counter() - t_kmodel

    summary = make_summary(y_true, pth_pred, onnx_pred, kmodel_pred)
    summary.update(selection)
    summary.update(
        {
            "raw_total_samples_before_selection": raw_total,
            "csv_file_count": int(len(set(source.tolist()))),
            "feature_mode": feature_mode,
            "infer_config": str(infer_config),
            "pth_dir": str(pth_dir),
            "pth_path": str(pth_path),
            "scaler_path": str(scaler_path),
            "onnx_path": str(onnx_path),
            "kmodel_path": str(kmodel_path),
            "data_dir": str(data_dir),
            "pth_infer_time_sec": float(pth_sec),
            "onnx_infer_time_sec": float(onnx_sec),
            "kmodel_infer_time_sec": float(kmodel_sec),
            "pipeline_total_time_sec": float(time.perf_counter() - t0),
            "details_csv": str(details_csv),
            "per_csv_csv": str(per_csv_csv),
            "per_dryness_csv": str(per_dryness_csv),
        }
    )

    save_rows_csv(
        details_csv,
        [
            "sample_id",
            "csv_name",
            "true_label",
            "pth_prediction",
            "onnx_prediction",
            "kmodel_prediction",
            "pth_abs_err",
            "onnx_abs_err",
            "kmodel_abs_err",
            "pth_onnx_abs_diff",
            "pth_kmodel_abs_diff",
            "onnx_kmodel_abs_diff",
        ],
        make_detail_rows(sample_ids, source, y_true, pth_pred, onnx_pred, kmodel_pred),
    )
    save_rows_csv(
        per_csv_csv,
        [
            "csv_name",
            "true_label",
            "samples",
            "pth_mae_vs_true",
            "onnx_mae_vs_true",
            "kmodel_mae_vs_true",
            "pth_vs_onnx_mae",
            "pth_vs_kmodel_mae",
            "onnx_vs_kmodel_mae",
            "onnx_vs_kmodel_max_abs",
        ],
        make_per_csv_rows(source, y_true, pth_pred, onnx_pred, kmodel_pred),
    )
    save_rows_csv(
        per_dryness_csv,
        [
            "dryness_label",
            "samples",
            "pth_mae_vs_true",
            "onnx_mae_vs_true",
            "kmodel_mae_vs_true",
            "pth_vs_onnx_mae",
            "pth_vs_kmodel_mae",
            "onnx_vs_kmodel_mae",
            "onnx_vs_kmodel_max_abs",
        ],
        make_per_dryness_rows(y_true, pth_pred, onnx_pred, kmodel_pred),
    )
    save_json(summary_json, summary)

    log("瀹屾垚")
    log("pth_mae_vs_true: {:.8f}".format(summary["pth_mae_vs_true"]))
    log("onnx_mae_vs_true: {:.8f}".format(summary["onnx_mae_vs_true"]))
    log("kmodel_mae_vs_true: {:.8f}".format(summary["kmodel_mae_vs_true"]))
    log("pth_vs_onnx_mae: {:.8f}".format(summary["pth_vs_onnx_mae"]))
    log("pth_vs_kmodel_mae: {:.8f}".format(summary["pth_vs_kmodel_mae"]))
    log("onnx_vs_kmodel_mae: {:.8f}".format(summary["onnx_vs_kmodel_mae"]))
    log("details_csv: {}".format(details_csv))
    log("per_csv_csv: {}".format(per_csv_csv))
    log("per_dryness_csv: {}".format(per_dryness_csv))
    log("summary_json: {}".format(summary_json))


class App:
    def __init__(self, root_window):
        self.root_window = root_window
        self.root_window.title("PTH / ONNX / KMODEL 瀵规瘮")
        self.project_root = Path(__file__).resolve().parent

        self.infer_config = StringVar(value=str((self.project_root / DEFAULT_INFER_CONFIG).resolve()))
        self.pth_dir = StringVar(value=str((self.project_root / "model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057").resolve()))
        self.onnx_path = StringVar(value=str((self.project_root / "../raw_cnn_k230/model/cnn_tcn_20260415_074057_i16u8_kld512.onnx").resolve()))
        self.kmodel_path = StringVar(value=str((self.project_root / "../raw_cnn_k230/model/cnn_tcn_20260415_074057_i16u8_kld512.kmodel").resolve()))
        self.data_dir = StringVar(value=str((self.project_root / "data/880k_data_260414").resolve()))
        self.output_dir = StringVar(value=str((self.project_root / DEFAULT_OUTPUT_DIR).resolve()))
        self.mode = StringVar(value="first_n")
        self.sample_value = StringVar(value="200")
        self.random_seed = StringVar(value="20260414")

        self._row = 0
        self.add_path_row("infer 閰嶇疆", self.infer_config, self.pick_json)
        self.add_path_row("pth 妯″瀷鐩綍", self.pth_dir, self.pick_dir)
        self.add_path_row("onnx 鏂囦欢", self.onnx_path, self.pick_onnx)
        self.add_path_row("kmodel 鏂囦欢", self.kmodel_path, self.pick_kmodel)
        self.add_path_row("娴嬭瘯鏁版嵁鐩綍", self.data_dir, self.pick_dir)
        self.add_path_row("杈撳嚭鐩綍", self.output_dir, self.pick_dir)

        Label(root_window, text="瀵规瘮鑼冨洿", width=14, anchor="e").grid(row=self._row, column=0, padx=6, pady=4)
        Radiobutton(root_window, text="鍏ㄩ噺", variable=self.mode, value="all").grid(row=self._row, column=1, sticky="w")
        self._row += 1
        Radiobutton(root_window, text="姣忎釜骞插害闅忔満鎶芥牱", variable=self.mode, value="random_per_dryness").grid(row=self._row, column=1, sticky="w")
        self._row += 1
        Radiobutton(root_window, text="前 N 个样本", variable=self.mode, value="first_n").grid(row=self._row, column=1, sticky="w")
        self._row += 1

        Label(root_window, text="N / 每干度数量", width=14, anchor="e").grid(row=self._row, column=0, padx=6, pady=4)
        Entry(root_window, textvariable=self.sample_value, width=18).grid(row=self._row, column=1, sticky="w", padx=6, pady=4)
        self._row += 1

        Label(root_window, text="闅忔満绉嶅瓙", width=14, anchor="e").grid(row=self._row, column=0, padx=6, pady=4)
        Entry(root_window, textvariable=self.random_seed, width=18).grid(row=self._row, column=1, sticky="w", padx=6, pady=4)
        self._row += 1

        self.run_button = Button(root_window, text="杩愯", command=self.start_run, width=16)
        self.run_button.grid(row=self._row, column=1, pady=8, sticky="w")
        self._row += 1

        self.log_box = Text(root_window, width=118, height=24)
        self.log_box.grid(row=self._row, column=0, columnspan=3, padx=8, pady=8)

    def add_path_row(self, title, variable, picker):
        Label(self.root_window, text=title, width=14, anchor="e").grid(row=self._row, column=0, padx=6, pady=4)
        Entry(self.root_window, textvariable=variable, width=100).grid(row=self._row, column=1, padx=6, pady=4)
        Button(self.root_window, text="閫夋嫨", command=lambda: picker(variable)).grid(row=self._row, column=2, padx=6, pady=4)
        self._row += 1

    def pick_json(self, variable):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def pick_onnx(self, variable):
        path = filedialog.askopenfilename(filetypes=[("ONNX", "*.onnx"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def pick_kmodel(self, variable):
        path = filedialog.askopenfilename(filetypes=[("KMODEL", "*.kmodel"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def pick_dir(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def log(self, text):
        def append():
            self.log_box.insert(END, str(text) + "\n")
            self.log_box.see(END)

        self.root_window.after(0, append)

    def start_run(self):
        self.run_button.config(state="disabled")
        self.log_box.delete("1.0", END)

        def worker():
            try:
                run_compare(
                    infer_config=Path(self.infer_config.get()).resolve(),
                    pth_dir=Path(self.pth_dir.get()).resolve(),
                    onnx_path=Path(self.onnx_path.get()).resolve(),
                    kmodel_path=Path(self.kmodel_path.get()).resolve(),
                    data_dir=Path(self.data_dir.get()).resolve(),
                    output_dir=Path(self.output_dir.get()).resolve(),
                    mode=self.mode.get(),
                    value=self.sample_value.get(),
                    seed=self.random_seed.get(),
                    log=self.log,
                )
                self.root_window.after(0, lambda: messagebox.showinfo("完成", "三模型对比完成"))
            except Exception:
                err = traceback.format_exc()
                self.log(err)
                self.root_window.after(0, lambda: messagebox.showerror("澶辫触", err))
            finally:
                self.root_window.after(0, lambda: self.run_button.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root_window = Tk()
    App(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
