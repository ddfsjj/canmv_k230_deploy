import argparse
import csv
import json
import os
import site
import time
from pathlib import Path

import joblib
import numpy as np
import torch

import infer

"""
PC 端 `.pth` 与 `kmodel` 对比脚本。

作用：
1. 用和 `infer.py` 一致的切窗逻辑构建样本。
2. 同一批样本先跑 `.pth`，再跑 `kmodel` 模拟器。
3. 生成汇总 JSON、逐 CSV 统计、逐干度统计、逐样本明细。
"""


def _bootstrap_nncase_env():
    # 在导入 nncase 前准备插件路径，避免首次导入时因为找不到 K230 插件而告警。
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PC .pth outputs and kmodel simulator outputs.")
    parser.add_argument("--infer_config", type=str, default="infer_config.json")
    parser.add_argument("--export_config", type=str, default="k230_export_config.json")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_dryness", type=int, default=None)
    parser.add_argument("--summary_json", type=str, default="compare_summary.json")
    parser.add_argument("--details_csv", type=str, default="compare_details.csv")
    parser.add_argument("--per_csv_csv", type=str, default="compare_per_csv.csv")
    parser.add_argument("--per_dryness_csv", type=str, default="compare_per_dryness.csv")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    return parser.parse_args()


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


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def prepare_nncase_env():
    # 保证本机模拟器能正确加载 K230 相关插件，不需要手工设环境变量。
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
    ordered = []
    for raw in site_roots:
        if not raw:
            continue
        norm = str(Path(raw).resolve())
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(Path(norm))

    plugin_dir = None
    package_root = None
    for root in ordered:
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


def resolve_output_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return (Path.cwd() / path).resolve()
    return (root / path).resolve()


def build_dataset_with_sources(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str,
    max_samples,
):
    # 和 infer.py 的切窗逻辑保持一致，只是额外保留每条样本来自哪个 CSV。
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
                X = np.stack(X_list).astype(np.float32)
                y = np.asarray(y_list, dtype=np.float32)
                source = np.asarray(source_list)
                return X, y, source

    if not X_list:
        return (
            np.empty((0, seq_length, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    source = np.asarray(source_list)
    return X, y, source


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


def run_pth_predictions(X_scaled: np.ndarray, infer_cfg: dict, root: Path, X_raw_aux=None):
    model_cfg = infer_cfg["model"]
    model_path = root / model_cfg["weights_path"]
    if not model_path.is_absolute():
        model_path = (root / model_cfg["weights_path"]).resolve()

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
    prepare_nncase_env()
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


def apply_sample_limits(
    X: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    max_samples,
    max_per_dryness,
):
    # 支持两层限流：
    # 1. 先按干度标签限制每个标签最多保留多少条样本，适合快速代表性测评。
    # 2. 再按全局 max_samples 截断，方便做更小规模的冒烟测试。
    keep_mask = np.ones((len(y),), dtype=bool)

    if max_per_dryness is not None:
        per_label_count = {}
        keep_mask[:] = False
        for idx, label in enumerate(y.tolist()):
            key = float(label)
            used = per_label_count.get(key, 0)
            if used >= max_per_dryness:
                continue
            keep_mask[idx] = True
            per_label_count[key] = used + 1

    X_sel = X[keep_mask]
    y_sel = y[keep_mask]
    source_sel = source[keep_mask]

    if max_samples is not None:
        limit = min(int(max_samples), int(X_sel.shape[0]))
        X_sel = X_sel[:limit]
        y_sel = y_sel[:limit]
        source_sel = source_sel[:limit]

    return X_sel, y_sel, source_sel


def apply_index_range(
    X: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    start_index,
    end_index,
):
    # 大将军，这里额外支持“只跑某一段样本”，方便并行分块对比。
    total = int(X.shape[0])
    start = int(start_index or 0)
    if start < 0 or start > total:
        raise IndexError(f"start_index out of range: {start}, total={total}")
    if end_index is None:
        end = total
    else:
        end = int(end_index)
    if end < start or end > total:
        raise IndexError(f"end_index out of range: start={start}, end={end}, total={total}")
    return X[start:end], y[start:end], source[start:end], start, end


def make_summary(y_true, pth_pred, kmodel_pred):
    pth_err = np.abs(pth_pred - y_true)
    k_err = np.abs(kmodel_pred - y_true)
    pth_vs_k = np.abs(kmodel_pred - pth_pred)
    return {
        "total_samples": int(len(y_true)),
        "nan_count": int(np.isnan(kmodel_pred).sum()),
        "pth_mae_vs_true": float(np.mean(pth_err)),
        "pth_rmse_vs_true": float(np.sqrt(np.mean((pth_pred - y_true) ** 2))),
        "kmodel_mae_vs_true": float(np.mean(k_err)),
        "kmodel_rmse_vs_true": float(np.sqrt(np.mean((kmodel_pred - y_true) ** 2))),
        "pth_vs_kmodel_mae": float(np.mean(pth_vs_k)),
        "pth_vs_kmodel_rmse": float(np.sqrt(np.mean((kmodel_pred - pth_pred) ** 2))),
        "pth_vs_kmodel_max_abs": float(np.max(pth_vs_k)),
        "pth_vs_kmodel_p95_abs": float(np.percentile(pth_vs_k, 95)),
        "pth_vs_kmodel_p99_abs": float(np.percentile(pth_vs_k, 99)),
    }


def make_per_csv_rows(source, y_true, pth_pred, kmodel_pred):
    rows = []
    for name in list(dict.fromkeys(source.tolist())):
        mask = source == name
        pth_diff = np.abs(pth_pred[mask] - y_true[mask])
        k_diff = np.abs(kmodel_pred[mask] - y_true[mask])
        pk_diff = np.abs(kmodel_pred[mask] - pth_pred[mask])
        rows.append(
            [
                name,
                int(np.sum(mask)),
                float(np.mean(pth_diff)),
                float(np.sqrt(np.mean((pth_pred[mask] - y_true[mask]) ** 2))),
                float(np.mean(k_diff)),
                float(np.sqrt(np.mean((kmodel_pred[mask] - y_true[mask]) ** 2))),
                float(np.mean(pk_diff)),
                float(np.sqrt(np.mean((kmodel_pred[mask] - pth_pred[mask]) ** 2))),
                float(np.max(pk_diff)),
            ]
        )
    rows.sort(key=lambda x: x[6], reverse=True)
    return rows


def make_per_dryness_rows(y_true, pth_pred, kmodel_pred):
    rows = []
    for value in sorted({float(v) for v in y_true.tolist()}):
        mask = y_true == np.float32(value)
        pth_diff = np.abs(pth_pred[mask] - y_true[mask])
        k_diff = np.abs(kmodel_pred[mask] - y_true[mask])
        pk_diff = np.abs(kmodel_pred[mask] - pth_pred[mask])
        rows.append(
            [
                float(value),
                int(np.sum(mask)),
                float(np.mean(pth_diff)),
                float(np.sqrt(np.mean((pth_pred[mask] - y_true[mask]) ** 2))),
                float(np.mean(k_diff)),
                float(np.sqrt(np.mean((kmodel_pred[mask] - y_true[mask]) ** 2))),
                float(np.mean(pk_diff)),
                float(np.sqrt(np.mean((kmodel_pred[mask] - pth_pred[mask]) ** 2))),
                float(np.max(pk_diff)),
            ]
        )
    rows.sort(key=lambda x: x[6], reverse=True)
    return rows


def make_detail_rows(source, y_true, pth_pred, kmodel_pred, sample_id_offset=0):
    diff = np.abs(kmodel_pred - pth_pred)
    rows = []
    for idx in range(len(y_true)):
        rows.append(
            [
                int(sample_id_offset + idx),
                str(source[idx]),
                float(y_true[idx]),
                float(pth_pred[idx]),
                float(kmodel_pred[idx]),
                float(diff[idx]),
            ]
        )
    rows.sort(key=lambda x: x[5], reverse=True)
    return rows


def main():
    # 这里固定先在 PC 侧把同一批样本喂给 `.pth` 和 `kmodel`，
    # 目的是排除板端串口因素，只看模型和量化本身的误差。
    args = parse_args()
    root = Path(__file__).resolve().parent
    infer_cfg_path = Path(args.infer_config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = root / infer_cfg_path
    export_cfg_path = Path(args.export_config)
    if not export_cfg_path.is_absolute():
        export_cfg_path = root / export_cfg_path

    infer_cfg = load_json(infer_cfg_path)
    export_cfg = load_json(export_cfg_path)
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError("data_dir not found: {}".format(data_dir))

    data_cfg = infer_cfg["data"]
    feature_mode = infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    max_samples = None if args.max_samples is None else require_positive_int(args.max_samples, "max_samples")
    max_per_dryness = (
        None if args.max_per_dryness is None else require_positive_int(args.max_per_dryness, "max_per_dryness")
    )
    if args.start_index is not None and int(args.start_index) < 0:
        raise ValueError("start_index must be >= 0")

    t_total_start = time.perf_counter()
    X, y, source = build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=feature_mode,
        max_samples=max_samples,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found under: {}".format(data_dir))
    raw_total_samples = int(X.shape[0])
    X_raw_aux = None
    if infer.normalize_model_type(infer_cfg["model"].get("type", "CNN-All")) == "cnn_tcn_seg3_soft_stats_moe":
        X_raw_aux, y_raw_aux, source_raw = build_seg3_raw_aux_with_sources(
            data_dir=data_dir,
            base_window_size=require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
            base_step=require_positive_int(data_cfg["base_step"], "data.base_step"),
            seq_length=require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
            seq_step=require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
            max_samples=max_samples,
        )
        if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y) or not np.array_equal(source_raw, source):
            raise RuntimeError("seg3 raw aux dataset is not aligned with main dataset.")

    X, y, source = apply_sample_limits(X, y, source, max_samples=max_samples, max_per_dryness=max_per_dryness)
    if X_raw_aux is not None:
        X_raw_aux, y_raw_aux, source_raw = apply_sample_limits(
            X_raw_aux,
            y_raw_aux,
            source_raw,
            max_samples=max_samples,
            max_per_dryness=max_per_dryness,
        )
        if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y) or not np.array_equal(source_raw, source):
            raise RuntimeError("seg3 raw aux dataset is not aligned after sample limits.")
    if X.shape[0] == 0:
        raise RuntimeError("No samples left after applying sample limits.")
    X, y, source, start_index, end_index = apply_index_range(X, y, source, args.start_index, args.end_index)
    if X_raw_aux is not None:
        X_raw_aux, y_raw_aux, source_raw, _, _ = apply_index_range(
            X_raw_aux,
            y_raw_aux,
            source_raw,
            args.start_index,
            args.end_index,
        )
        if X_raw_aux.shape != X.shape or not np.array_equal(y_raw_aux, y) or not np.array_equal(source_raw, source):
            raise RuntimeError("seg3 raw aux dataset is not aligned after index range.")
    if X.shape[0] == 0:
        raise RuntimeError("No samples left after applying index range.")

    scaler_path = root / infer_cfg["normalization"]["scaler_path"]
    scaler_path = scaler_path.resolve()
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    t_pth_start = time.perf_counter()
    pth_pred = run_pth_predictions(X_scaled, infer_cfg, root, X_raw_aux=X_raw_aux)
    t_pth_end = time.perf_counter()

    kmodel_path = root / export_cfg["paths"]["kmodel"]
    kmodel_path = kmodel_path.resolve()
    t_k_start = time.perf_counter()
    kmodel_pred = run_kmodel_predictions(X_scaled, kmodel_path, args.log_every, X_raw_aux=X_raw_aux)
    t_k_end = time.perf_counter()

    summary = make_summary(y, pth_pred, kmodel_pred)
    summary["data_dir"] = str(data_dir)
    summary["csv_file_count"] = len(sorted(data_dir.glob("*.csv")))
    summary["feature_mode"] = feature_mode
    summary["raw_total_samples_before_limit"] = raw_total_samples
    summary["slice_start_index"] = int(start_index)
    summary["slice_end_index"] = int(end_index)
    summary["max_samples"] = max_samples
    summary["max_per_dryness"] = max_per_dryness
    summary["pth_infer_time_sec"] = float(t_pth_end - t_pth_start)
    summary["kmodel_infer_time_sec"] = float(t_k_end - t_k_start)
    summary["pipeline_total_time_sec"] = float(time.perf_counter() - t_total_start)

    per_csv_rows = make_per_csv_rows(source, y, pth_pred, kmodel_pred)
    per_dryness_rows = make_per_dryness_rows(y, pth_pred, kmodel_pred)
    detail_rows = make_detail_rows(source, y, pth_pred, kmodel_pred, sample_id_offset=start_index)

    summary_json = resolve_output_path(args.summary_json, root)
    details_csv = resolve_output_path(args.details_csv, root)
    per_csv_csv = resolve_output_path(args.per_csv_csv, root)
    per_dryness_csv = resolve_output_path(args.per_dryness_csv, root)

    save_json(summary_json, summary)
    save_rows_csv(
        per_csv_csv,
        [
            "csv_name",
            "samples",
            "pth_mae_vs_true",
            "pth_rmse_vs_true",
            "kmodel_mae_vs_true",
            "kmodel_rmse_vs_true",
            "pth_vs_kmodel_mae",
            "pth_vs_kmodel_rmse",
            "pth_vs_kmodel_max_abs",
        ],
        per_csv_rows,
    )
    save_rows_csv(
        per_dryness_csv,
        [
            "dryness_label",
            "samples",
            "pth_mae_vs_true",
            "pth_rmse_vs_true",
            "kmodel_mae_vs_true",
            "kmodel_rmse_vs_true",
            "pth_vs_kmodel_mae",
            "pth_vs_kmodel_rmse",
            "pth_vs_kmodel_max_abs",
        ],
        per_dryness_rows,
    )
    save_rows_csv(
        details_csv,
        ["sample_id", "csv_name", "true_label", "pth_prediction", "kmodel_prediction", "abs_diff"],
        detail_rows,
    )

    print("=== PTH vs KMODEL Compare ===")
    print("data_dir:", data_dir)
    print("csv_file_count:", summary["csv_file_count"])
    print("raw_total_samples_before_limit:", summary["raw_total_samples_before_limit"])
    print("total_samples:", summary["total_samples"])
    print("slice_start_index:", summary["slice_start_index"])
    print("slice_end_index:", summary["slice_end_index"])
    print("max_samples:", summary["max_samples"])
    print("max_per_dryness:", summary["max_per_dryness"])
    print("nan_count:", summary["nan_count"])
    print("pth_mae_vs_true:", summary["pth_mae_vs_true"])
    print("pth_rmse_vs_true:", summary["pth_rmse_vs_true"])
    print("kmodel_mae_vs_true:", summary["kmodel_mae_vs_true"])
    print("kmodel_rmse_vs_true:", summary["kmodel_rmse_vs_true"])
    print("pth_vs_kmodel_mae:", summary["pth_vs_kmodel_mae"])
    print("pth_vs_kmodel_rmse:", summary["pth_vs_kmodel_rmse"])
    print("pth_vs_kmodel_max_abs:", summary["pth_vs_kmodel_max_abs"])
    print("pth_vs_kmodel_p95_abs:", summary["pth_vs_kmodel_p95_abs"])
    print("pth_vs_kmodel_p99_abs:", summary["pth_vs_kmodel_p99_abs"])
    print("pth_infer_time_sec:", summary["pth_infer_time_sec"])
    print("kmodel_infer_time_sec:", summary["kmodel_infer_time_sec"])
    print("pipeline_total_time_sec:", summary["pipeline_total_time_sec"])
    print("summary_json:", summary_json)
    print("per_csv_csv:", per_csv_csv)
    print("per_dryness_csv:", per_dryness_csv)
    print("details_csv:", details_csv)
    print("top_10_per_csv:")
    for row in per_csv_rows[:10]:
        print(
            "csv={}, samples={}, pth_mae={:.6f}, pth_rmse={:.6f}, kmodel_mae={:.6f}, kmodel_rmse={:.6f}, pth_vs_k_mae={:.6f}, pth_vs_k_rmse={:.6f}, pth_vs_k_max={:.6f}".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
            )
        )
    print("per_dryness:")
    for row in per_dryness_rows:
        print(
            "dryness={:.12f}, samples={}, pth_mae={:.6f}, pth_rmse={:.6f}, kmodel_mae={:.6f}, kmodel_rmse={:.6f}, pth_vs_k_mae={:.6f}, pth_vs_k_rmse={:.6f}, pth_vs_k_max={:.6f}".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
            )
        )
    print("top_20_abs_diff:")
    for row in detail_rows[:20]:
        print(
            "idx={}, csv={}, true={:.6f}, pth={:.6f}, kmodel={:.6f}, abs_diff={:.6f}".format(
                row[0], row[1], row[2], row[3], row[4], row[5]
            )
        )


if __name__ == "__main__":
    main()
