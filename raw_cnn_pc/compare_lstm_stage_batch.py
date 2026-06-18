import argparse
import csv
import json
from pathlib import Path

import numpy as np

import compare_lstm_cutoff_pth_kmodel as cutoff_debug
import compare_pth_kmodel


"""
澶у皢鍐涳紝杩欎釜鑴氭湰鐜板湪浣滀负鍥哄畾璇勬祴鑴氭湰浣跨敤銆?
鍥哄畾鍙ｅ緞濡備笅锛?1. 姣忔閮藉悓鏃惰瘎浼颁笁缁勮宸細
   - `lstm_input`锛氳繘鍏?GNNELSTM 鍓嶇殑 `%33`
   - `lstm_last`锛歀STM 鏈€鍚庝竴涓椂闂存杈撳嚭 `Y[-1]`
   - `scalar_output`锛氭渶缁堟爣閲忚緭鍑?2. 姣忔閮藉浐瀹氭寜涓夌粍鏍锋湰缁熻锛?   - `all`
   - `high_dryness`
   - `non_high_dryness`
3. 鍚庣画鎵€鏈夊疄楠岀粺涓€寮曠敤 `batch_summary.json` 閲岀殑杩欏缁撴瀯銆?
杈撳嚭鏂囦欢锛?1. `batch_summary.json`锛氬浐瀹氭眹鎬荤粨鏋?2. `lstm_input_per_sample.csv`
3. `lstm_last_per_sample.csv`
4. `scalar_output_per_sample.csv`
"""


def parse_args():
    parser = argparse.ArgumentParser(description="鍥哄畾鍙ｅ緞璇勬祴锛歭stm_input / lstm_last / scalar_output")
    parser.add_argument("--infer_config", type=str, required=True)
    parser.add_argument("--export_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/layerwise/lstm_stage_batch")
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--high_dryness_threshold", type=float, default=0.75)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    return parser.parse_args()


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def resolve_cli_path(root: Path, raw_path: str, kind: str) -> Path:
    # 优先按当前工作目录解析，找不到时再回退到 raw_cnn_pc 根目录。
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or kind == "output":
        return cwd_candidate
    return (root / path).resolve()


def compute_tensor_stats(diff_nd: np.ndarray):
    flat = diff_nd.reshape(diff_nd.shape[0], -1)
    mae_per_sample = np.mean(np.abs(flat), axis=1)
    rmse_per_sample = np.sqrt(np.mean(flat ** 2, axis=1))
    max_abs_per_sample = np.max(np.abs(flat), axis=1)
    return {
        "sample_count": int(flat.shape[0]),
        "mae_mean": float(np.mean(mae_per_sample)),
        "rmse_mean": float(np.mean(rmse_per_sample)),
        "max_abs_max": float(np.max(max_abs_per_sample)),
    }


def compute_scalar_stats(pred: np.ndarray, truth: np.ndarray):
    pred = pred.reshape(-1).astype(np.float32)
    truth = truth.reshape(-1).astype(np.float32)
    diff = pred - truth
    return {
        "sample_count": int(pred.shape[0]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(truth)),
    }


def select_subset(X_scaled: np.ndarray, y: np.ndarray, source: np.ndarray, sample_count: int):
    total = int(X_scaled.shape[0])
    count = min(int(sample_count), total)
    indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
    return X_scaled[indices], y[indices], source[indices], indices


def stage_compare(
    stage: str,
    scheme_name: str,
    root: Path,
    infer_cfg: dict,
    export_cfg: dict,
    X_subset: np.ndarray,
    y_subset: np.ndarray,
    source_subset: np.ndarray,
    output_dir: Path,
    max_calib_samples,
):
    input_shape = tuple(X_subset.shape[1:])
    base_model = cutoff_debug.build_debug_model(root=root, infer_cfg=infer_cfg, input_shape=input_shape)
    if stage == "lstm_input":
        debug_model = cutoff_debug.LSTMInputDebug(base_model).eval()
    else:
        debug_model = cutoff_debug.LSTMLastStepDebug(base_model).eval()

    cutoff_cfg = cutoff_debug.clone_export_cfg_for_cutoff(
        export_cfg=export_cfg,
        root=root,
        output_dir=output_dir,
        scheme_name=scheme_name,
    )
    cutoff_debug.export_cutoff_onnx(debug_model, Path(cutoff_cfg["paths"]["onnx"]), sample_shape=input_shape)

    scaler_pkl = Path(export_cfg["paths"]["scaler_pkl"])
    if not scaler_pkl.is_absolute():
        scaler_pkl = (root / scaler_pkl).resolve()
    cutoff_debug.build_kmodel.export_scaler_json(scaler_pkl, Path(cutoff_cfg["paths"]["scaler_json"]))

    calib_count = (
        int(max_calib_samples)
        if max_calib_samples is not None
        else cutoff_debug.build_kmodel.resolve_calibration_sample_count(
            export_cfg["quantization"].get("samples_count", 64),
            X_subset.shape[0],
            "quantization.samples_count",
        )
    )
    calibration_data = cutoff_debug.build_kmodel.select_calibration_data(
        X_scaled=X_subset,
        count=calib_count,
        strategy=export_cfg["quantization"].get("sampling_strategy", "first"),
        random_seed=export_cfg["quantization"].get("random_seed", None),
        y_labels=y_subset,
    )
    np.save(Path(cutoff_cfg["paths"]["calibration_npy"]), calibration_data.astype(np.float32))
    cutoff_debug.build_kmodel.compile_kmodel_with_nncase(
        cutoff_cfg,
        root=root,
        calibration_data=calibration_data.astype(np.float32),
    )

    pth_outputs = []
    kmodel_outputs = []
    for i in range(int(X_subset.shape[0])):
        sample = X_subset[i : i + 1].astype(np.float32)
        pth_outputs.append(cutoff_debug.run_pth_last_y(debug_model, sample))
        kmodel_outputs.append(cutoff_debug.run_kmodel_last_y(sample, Path(cutoff_cfg["paths"]["kmodel"])))

    pth_outputs = np.concatenate(pth_outputs, axis=0).astype(np.float32)
    kmodel_outputs = np.concatenate(kmodel_outputs, axis=0).astype(np.float32)
    diff = (kmodel_outputs - pth_outputs).astype(np.float32)

    flat = diff.reshape(diff.shape[0], -1)
    rows = []
    for i in range(int(X_subset.shape[0])):
        rows.append(
            [
                int(i),
                str(source_subset[i]),
                float(y_subset[i]),
                float(np.mean(np.abs(flat[i]))),
                float(np.sqrt(np.mean(flat[i] ** 2))),
                float(np.max(np.abs(flat[i]))),
            ]
        )

    return {
        "pth_outputs": pth_outputs,
        "kmodel_outputs": kmodel_outputs,
        "diff": diff,
        "per_sample_rows": rows,
        "cutoff_kmodel_path": str(Path(cutoff_cfg["paths"]["kmodel"]).resolve()),
    }


def scalar_compare(root: Path, infer_cfg: dict, export_cfg: dict, X_subset: np.ndarray, y_subset: np.ndarray, source_subset: np.ndarray):
    pth_preds = compare_pth_kmodel.run_pth_predictions(X_subset.astype(np.float32), infer_cfg, root)
    kmodel_path = Path(export_cfg["paths"]["kmodel"])
    if not kmodel_path.is_absolute():
        kmodel_path = (root / export_cfg["paths"]["kmodel"]).resolve()
    kmodel_preds = compare_pth_kmodel.run_kmodel_predictions(X_subset.astype(np.float32), kmodel_path, log_every=0)

    rows = []
    for i in range(int(X_subset.shape[0])):
        abs_err = abs(float(kmodel_preds[i] - y_subset[i]))
        rows.append(
            [
                int(i),
                str(source_subset[i]),
                float(y_subset[i]),
                float(pth_preds[i]),
                float(kmodel_preds[i]),
                float(abs_err),
                float(abs_err),
            ]
        )
    return {
        "pth_preds": pth_preds.astype(np.float32),
        "kmodel_preds": kmodel_preds.astype(np.float32),
        "per_sample_rows": rows,
        "kmodel_path": str(kmodel_path.resolve()),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    infer_cfg_path = resolve_cli_path(root, args.infer_config, "input")
    export_cfg_path = resolve_cli_path(root, args.export_config, "input")
    data_dir = resolve_cli_path(root, args.data_dir, "input")
    output_dir = resolve_cli_path(root, args.output_dir, "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = cutoff_debug.load_json(infer_cfg_path)
    export_cfg = cutoff_debug.load_json(export_cfg_path)

    data_cfg = infer_cfg["data"]
    X, y, source = compare_pth_kmodel.build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=compare_pth_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=compare_pth_kmodel.require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=compare_pth_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=compare_pth_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=cutoff_debug.infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw")),
        max_samples=None,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"娌℃湁浠庢暟鎹洰褰曟瀯寤哄嚭鏍锋湰: {data_dir}")

    scaler_path = (root / infer_cfg["normalization"]["scaler_path"]).resolve()
    scaler = cutoff_debug.joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    X_subset, y_subset, source_subset, indices = select_subset(
        X_scaled=X_scaled,
        y=y.astype(np.float32),
        source=source,
        sample_count=int(args.sample_count),
    )

    lstm_input_result = stage_compare(
        stage="lstm_input",
        scheme_name="batch_lstm_input",
        root=root,
        infer_cfg=infer_cfg,
        export_cfg=export_cfg,
        X_subset=X_subset,
        y_subset=y_subset,
        source_subset=source_subset,
        output_dir=output_dir / "lstm_input",
        max_calib_samples=args.max_calib_samples,
    )
    lstm_last_result = stage_compare(
        stage="lstm_last",
        scheme_name="batch_lstm_last",
        root=root,
        infer_cfg=infer_cfg,
        export_cfg=export_cfg,
        X_subset=X_subset,
        y_subset=y_subset,
        source_subset=source_subset,
        output_dir=output_dir / "lstm_last",
        max_calib_samples=args.max_calib_samples,
    )
    scalar_result = scalar_compare(
        root=root,
        infer_cfg=infer_cfg,
        export_cfg=export_cfg,
        X_subset=X_subset,
        y_subset=y_subset,
        source_subset=source_subset,
    )

    high_mask = y_subset >= float(args.high_dryness_threshold)
    low_mask = np.logical_not(high_mask)

    summary = {
        "evaluation_protocol": {
            "version": "fixed_v1",
            "stages": ["lstm_input", "lstm_last", "scalar_output"],
            "groups": ["all", "high_dryness", "non_high_dryness"],
        },
        "sample_count": int(X_subset.shape[0]),
        "selected_indices": indices.tolist(),
        "high_dryness_threshold": float(args.high_dryness_threshold),
        "high_dryness_count": int(np.sum(high_mask)),
        "non_high_dryness_count": int(np.sum(low_mask)),
        "grouped_metrics": {
            "all": {
                "lstm_input": compute_tensor_stats(lstm_input_result["diff"]),
                "lstm_last": compute_tensor_stats(lstm_last_result["diff"]),
                "scalar_output": compute_scalar_stats(scalar_result["kmodel_preds"], y_subset),
            },
            "high_dryness": {
                "lstm_input": compute_tensor_stats(lstm_input_result["diff"][high_mask]) if np.any(high_mask) else None,
                "lstm_last": compute_tensor_stats(lstm_last_result["diff"][high_mask]) if np.any(high_mask) else None,
                "scalar_output": compute_scalar_stats(scalar_result["kmodel_preds"][high_mask], y_subset[high_mask]) if np.any(high_mask) else None,
            },
            "non_high_dryness": {
                "lstm_input": compute_tensor_stats(lstm_input_result["diff"][low_mask]) if np.any(low_mask) else None,
                "lstm_last": compute_tensor_stats(lstm_last_result["diff"][low_mask]) if np.any(low_mask) else None,
                "scalar_output": compute_scalar_stats(scalar_result["kmodel_preds"][low_mask], y_subset[low_mask]) if np.any(low_mask) else None,
            },
        },
        "lstm_input_kmodel_path": lstm_input_result["cutoff_kmodel_path"],
        "lstm_last_kmodel_path": lstm_last_result["cutoff_kmodel_path"],
        "scalar_output_kmodel_path": scalar_result["kmodel_path"],
    }

    save_json(output_dir / "batch_summary.json", summary)
    save_csv(
        output_dir / "lstm_input_per_sample.csv",
        ["subset_idx", "csv_name", "true_label", "mae", "rmse", "max_abs"],
        lstm_input_result["per_sample_rows"],
    )
    save_csv(
        output_dir / "lstm_last_per_sample.csv",
        ["subset_idx", "csv_name", "true_label", "mae", "rmse", "max_abs"],
        lstm_last_result["per_sample_rows"],
    )
    save_csv(
        output_dir / "scalar_output_per_sample.csv",
        ["subset_idx", "csv_name", "true_label", "pth_pred", "kmodel_pred", "abs_err", "rmse_like_abs"],
        scalar_result["per_sample_rows"],
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
