import argparse
import json
from pathlib import Path

import numpy as np
import torch

import build_kmodel
import compare_lstm_cutoff_pth_kmodel as cutoff_debug
import compare_pth_kmodel
import infer


"""
澶у皢鍐涳紝杩欎釜鑴氭湰鐢ㄤ簬鍥哄畾涓€鐗?ONNX 鍚庯紝璺戜竴涓?3脳2 鐨勯噺鍖栧疄楠岀煩闃点€?
鍥哄畾鍐呭锛?1. 鍚屼竴浠?full ONNX锛氱敤浜庢渶缁堟爣閲忚緭鍑虹殑 kmodel
2. 鍚屼竴浠?lstm_last ONNX锛氱敤浜?Y[-1] 杈撳嚭鐨?kmodel
3. 鍚屼竴鎵硅瘎娴嬫牱鏈€夋嫨瑙勫垯锛氬叏閲忔牱鏈寜鍥哄畾绱㈠紩鎶芥牱锛屽啀浠庝腑鍒囧嚭楂樺共搴﹀瓙闆?
鍙樺寲鍐呭锛?1. 鏍″噯闆嗙瓥鐣ワ細
   - random
   - balanced锛堟寜骞插害鍧囪　鎶芥牱锛?   - high_dryness_weighted锛堥珮骞插害鍔犳潈锛?2. 閲忓寲绮惧害锛?   - int8
   - int16_pref锛堣繖閲屽畾涔変负婵€娲?int16銆佹潈閲?uint8锛?
杈撳嚭閲嶇偣锛?1. 姣忎釜鏍煎瓙閮戒繚瀛?summary.json
2. 鎬昏〃鍙眹鎬婚珮骞插害锛?   - lstm_last MAE / RMSE
   - scalar_output MAE / RMSE
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed ONNX 3x2 quantization matrix evaluation.")
    parser.add_argument("--infer_config", type=str, required=True)
    parser.add_argument("--export_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/quantization/fixed_onnx_matrix")
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--high_dryness_threshold", type=float, default=0.75)
    parser.add_argument("--calibration_samples", type=int, default=1024)
    parser.add_argument("--random_seed", type=int, default=20260413)
    parser.add_argument("--only_calibration_strategy", type=str, default=None)
    parser.add_argument("--only_precision_scheme", type=str, default=None)
    return parser.parse_args()


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_cli_path(root: Path, raw_path: str, kind: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or kind == "output":
        return cwd_candidate
    return (root / path).resolve()


def select_eval_subset(X_scaled: np.ndarray, y: np.ndarray, source: np.ndarray, sample_count: int):
    total = int(X_scaled.shape[0])
    count = min(int(sample_count), total)
    indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
    return X_scaled[indices], y[indices], source[indices], indices


def select_high_dryness_weighted_indices(y_labels: np.ndarray, count: int, threshold: float) -> np.ndarray:
    # 澶у皢鍐涳紝杩欓噷瀹氫箟鈥滈珮骞插害鍔犳潈鈥濓細
    # 1. 70% 閰嶉缁欓珮骞插害鏍锋湰
    # 2. 30% 配额给非高干度样本。
    # 3. 每个子集内部再按干度均匀抽样，避免只取到某一个文件。
    total = int(y_labels.shape[0])
    if count >= total:
        return np.arange(total, dtype=np.int64)

    high_idx = np.flatnonzero(y_labels >= np.float32(threshold)).astype(np.int64)
    low_idx = np.flatnonzero(y_labels < np.float32(threshold)).astype(np.int64)

    high_quota = int(round(count * 0.7))
    low_quota = int(count - high_quota)

    high_quota = min(high_quota, int(high_idx.size))
    low_quota = min(low_quota, int(low_idx.size))

    # 如果某一边不够，就把剩余额度补给另一边。
    remaining = int(count - high_quota - low_quota)
    if remaining > 0:
        if int(high_idx.size) - high_quota >= int(low_idx.size) - low_quota:
            high_quota = min(int(high_idx.size), high_quota + remaining)
        else:
            low_quota = min(int(low_idx.size), low_quota + remaining)

    parts = []
    if high_quota > 0:
        high_local = build_kmodel._select_per_dryness_uniform_indices(y_labels[high_idx], high_quota)
        parts.append(high_idx[high_local])
    if low_quota > 0:
        low_local = build_kmodel._select_per_dryness_uniform_indices(y_labels[low_idx], low_quota)
        parts.append(low_idx[low_local])

    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.concatenate(parts).astype(np.int64))


def select_calibration_by_strategy(X_scaled: np.ndarray, y_labels: np.ndarray, count: int, strategy: str, random_seed: int, threshold: float):
    if strategy == "random":
        return build_kmodel.select_calibration_data(
            X_scaled=X_scaled,
            count=count,
            strategy="random",
            random_seed=random_seed,
            y_labels=y_labels,
        )
    if strategy == "balanced":
        return build_kmodel.select_calibration_data(
            X_scaled=X_scaled,
            count=count,
            strategy="per_dryness_uniform",
            random_seed=random_seed,
            y_labels=y_labels,
        )
    if strategy == "high_dryness_weighted":
        indices = select_high_dryness_weighted_indices(y_labels, count, threshold)
        if indices.size == 0:
            raise RuntimeError("高干度加权抽样没有选出任何校准样本。")
        return X_scaled[indices].astype(np.float32)
    raise ValueError(f"鏈煡鏍″噯绛栫暐: {strategy}")


def export_fixed_models(root: Path, infer_cfg: dict, export_cfg: dict, output_dir: Path, input_shape):
    model_cfg = infer_cfg["model"]
    weights_path = (root / model_cfg["weights_path"]).resolve()
    state_dict = infer.load_state_dict_compat(weights_path, torch.device("cpu"))
    base_model = infer.build_model_from_config(model_cfg=model_cfg, input_shape=input_shape, state_dict=state_dict)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    full_onnx = output_dir / "fixed_full.onnx"
    build_kmodel.export_onnx(base_model, full_onnx, input_shape=input_shape)

    lstm_last_model = cutoff_debug.LSTMLastStepDebug(base_model).eval()
    lstm_last_onnx = output_dir / "fixed_lstm_last.onnx"
    cutoff_debug.export_cutoff_onnx(lstm_last_model, lstm_last_onnx, sample_shape=input_shape)

    scaler_pkl = Path(export_cfg["paths"]["scaler_pkl"])
    if not scaler_pkl.is_absolute():
        scaler_pkl = (root / scaler_pkl).resolve()
    scaler_json = output_dir / "fixed_scaler.json"
    build_kmodel.export_scaler_json(scaler_pkl, scaler_json)

    return {
        "base_model": base_model,
        "full_onnx": full_onnx.resolve(),
        "lstm_last_onnx": lstm_last_onnx.resolve(),
        "scaler_json": scaler_json.resolve(),
    }


def make_scheme_cfg(base_export_cfg: dict, onnx_path: Path, scaler_json: Path, output_dir: Path, scheme_id: str, quant_type: str, weight_quant_type: str):
    cfg = json.loads(json.dumps(base_export_cfg))
    cfg["name"] = scheme_id
    cfg["paths"]["onnx"] = str(onnx_path.resolve())
    cfg["paths"]["scaler_json"] = str(scaler_json.resolve())
    cfg["paths"]["kmodel"] = str((output_dir / f"{scheme_id}.kmodel").resolve())
    cfg["paths"]["calibration_npy"] = str((output_dir / f"{scheme_id}_calibration.npy").resolve())
    cfg["paths"]["nncase_dump_dir"] = str((output_dir / f"{scheme_id}_nncase_dump").resolve())
    cfg["quantization"]["quant_type"] = quant_type
    cfg["quantization"]["weight_quant_type"] = weight_quant_type
    return cfg


def compute_tensor_stats(pred: np.ndarray, truth: np.ndarray):
    diff = pred.astype(np.float32) - truth.astype(np.float32)
    flat = diff.reshape(diff.shape[0], -1)
    return {
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(flat ** 2))),
    }


def compute_scalar_stats(pred: np.ndarray, truth: np.ndarray):
    diff = pred.reshape(-1).astype(np.float32) - truth.reshape(-1).astype(np.float32)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def run_simulator_outputs(X_subset: np.ndarray, kmodel_path: Path):
    compare_pth_kmodel.prepare_nncase_env()
    import nncase

    sim = nncase.Simulator()
    sim.load_model(kmodel_path.read_bytes())
    outputs = []
    for i in range(int(X_subset.shape[0])):
        sample = X_subset[i : i + 1].astype(np.float32)
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(sample))
        sim.run()
        outputs.append(np.asarray(sim.get_output_tensor(0).to_numpy(), dtype=np.float32))
    return np.concatenate(outputs, axis=0).astype(np.float32)


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
        feature_mode=infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw")),
        max_samples=None,
    )
    scaler_path = (root / infer_cfg["normalization"]["scaler_path"]).resolve()
    scaler = cutoff_debug.joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    y = y.astype(np.float32)

    X_subset, y_subset, source_subset, selected_indices = select_eval_subset(
        X_scaled=X_scaled,
        y=y,
        source=source,
        sample_count=int(args.sample_count),
    )
    high_mask = y_subset >= float(args.high_dryness_threshold)

    fixed = export_fixed_models(
        root=root,
        infer_cfg=infer_cfg,
        export_cfg=export_cfg,
        output_dir=output_dir / "fixed_onnx",
        input_shape=tuple(X_subset.shape[1:]),
    )

    pth_scalar = compare_pth_kmodel.run_pth_predictions(X_subset.astype(np.float32), infer_cfg, root)
    pth_lstm_last = []
    lstm_last_debug_model = cutoff_debug.LSTMLastStepDebug(fixed["base_model"]).eval()
    for i in range(int(X_subset.shape[0])):
        pth_lstm_last.append(cutoff_debug.run_pth_last_y(lstm_last_debug_model, X_subset[i : i + 1]))
    pth_lstm_last = np.concatenate(pth_lstm_last, axis=0).astype(np.float32)

    matrix = []
    calibration_strategies = [
        ("random", "random"),
        ("balanced", "balanced"),
        ("high_dryness_weighted", "high_dryness_weighted"),
    ]
    precision_schemes = [
        ("int8", "int8", "int8"),
        ("int16_pref", "int16", "uint8"),
    ]

    if args.only_calibration_strategy:
        calibration_strategies = [
            item for item in calibration_strategies if item[0] == str(args.only_calibration_strategy)
        ]
        if not calibration_strategies:
            raise ValueError(f"鏈煡鏍″噯绛栫暐: {args.only_calibration_strategy}")

    if args.only_precision_scheme:
        precision_schemes = [
            item for item in precision_schemes if item[0] == str(args.only_precision_scheme)
        ]
        if not precision_schemes:
            raise ValueError(f"鏈煡閲忓寲绮惧害鏂规: {args.only_precision_scheme}")

    for strategy_id, strategy_name in calibration_strategies:
        calibration_data = select_calibration_by_strategy(
            X_scaled=X_scaled,
            y_labels=y,
            count=int(args.calibration_samples),
            strategy=strategy_name,
            random_seed=int(args.random_seed),
            threshold=float(args.high_dryness_threshold),
        )

        for precision_id, quant_type, weight_quant_type in precision_schemes:
            scheme_id = f"{strategy_id}__{precision_id}"
            scheme_dir = output_dir / scheme_id
            scheme_dir.mkdir(parents=True, exist_ok=True)

            full_cfg = make_scheme_cfg(
                base_export_cfg=export_cfg,
                onnx_path=fixed["full_onnx"],
                scaler_json=fixed["scaler_json"],
                output_dir=scheme_dir,
                scheme_id=f"{scheme_id}_full",
                quant_type=quant_type,
                weight_quant_type=weight_quant_type,
            )
            np.save(Path(full_cfg["paths"]["calibration_npy"]), calibration_data.astype(np.float32))
            build_kmodel.compile_kmodel_with_nncase(full_cfg, root=root, calibration_data=calibration_data.astype(np.float32))

            lstm_cfg = make_scheme_cfg(
                base_export_cfg=export_cfg,
                onnx_path=fixed["lstm_last_onnx"],
                scaler_json=fixed["scaler_json"],
                output_dir=scheme_dir,
                scheme_id=f"{scheme_id}_lstm_last",
                quant_type=quant_type,
                weight_quant_type=weight_quant_type,
            )
            np.save(Path(lstm_cfg["paths"]["calibration_npy"]), calibration_data.astype(np.float32))
            build_kmodel.compile_kmodel_with_nncase(lstm_cfg, root=root, calibration_data=calibration_data.astype(np.float32))

            kmodel_scalar = run_simulator_outputs(X_subset, Path(full_cfg["paths"]["kmodel"]))
            kmodel_lstm_last = run_simulator_outputs(X_subset, Path(lstm_cfg["paths"]["kmodel"]))

            high_lstm_stats = compute_tensor_stats(kmodel_lstm_last[high_mask], pth_lstm_last[high_mask])
            high_scalar_stats = compute_scalar_stats(kmodel_scalar[high_mask], y_subset[high_mask])

            result = {
                "scheme_id": scheme_id,
                "calibration_strategy": strategy_id,
                "precision_scheme": precision_id,
                "quant_type": quant_type,
                "weight_quant_type": weight_quant_type,
                "high_dryness_threshold": float(args.high_dryness_threshold),
                "high_dryness_count": int(np.sum(high_mask)),
                "high_dryness_lstm_last": high_lstm_stats,
                "high_dryness_scalar_output": high_scalar_stats,
                "full_kmodel_path": str(Path(full_cfg["paths"]["kmodel"]).resolve()),
                "lstm_last_kmodel_path": str(Path(lstm_cfg["paths"]["kmodel"]).resolve()),
            }
            save_json(scheme_dir / "summary.json", result)
            matrix.append(result)

    summary = {
        "evaluation_protocol": {
            "fixed_full_onnx": str(fixed["full_onnx"]),
            "fixed_lstm_last_onnx": str(fixed["lstm_last_onnx"]),
            "sample_count": int(X_subset.shape[0]),
            "selected_indices": selected_indices.tolist(),
            "high_dryness_threshold": float(args.high_dryness_threshold),
            "high_dryness_count": int(np.sum(high_mask)),
        },
        "matrix_results": matrix,
    }
    save_json(output_dir / "matrix_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
