import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

import build_kmodel
import compare_pth_kmodel
import infer

"""
澶у皢鍐涳紝杩欎釜鑴氭湰涓撻棬鐢ㄤ簬楠岃瘉锛?1. 鍚屼竴鏍锋湰鍦?LSTM 鏈€鍚庝竴涓椂闂存杈撳嚭 `Y[-1]` 澶勶紝`.pth` 鍜?`kmodel simulator` 鐨勮宸湁澶氬ぇ銆?2. 璇樊鏄笉鏄湪 LSTM 杈撳嚭杩欎竴灞傚氨宸茬粡绐佺劧鏀惧ぇ銆?
鑴氭湰鍋氭硶锛?1. 璇诲彇姝ｅ紡 infer/export 閰嶇疆锛屾寜姝ｅ紡棰勫鐞嗛€昏緫鏋勫缓鏍锋湰銆?2. 閫夊彇鍚屼竴涓牱鏈紝鍚屾椂璺戯細
   - `.pth` 鎴柇妯″瀷锛氳緭鍑?`Y[-1]`锛屽舰鐘?`[1, 50]`
   - `kmodel` 鎴柇妯″瀷锛氬悓鏍疯緭鍑?`Y[-1]`
3. 杈撳嚭閫愮淮璇樊銆丩2/L1 缁熻锛屽苟椤烘墜缁欏嚭鍚屼竴鏍锋湰鏈€缁堟爣閲忚緭鍑虹殑璇樊浣滀负鍙傜収銆?
娉ㄦ剰锛?1. 杩欓噷瀵煎嚭鐨勪笉鏄渶缁堝洖褰掑ご妯″瀷锛岃€屾槸鈥滄埅鏂埌 LSTM 鏈€鍚庝竴姝ヨ緭鍑衡€濈殑璋冭瘯 kmodel銆?2. 杩欐牱鍙互鐩存帴鍒ゆ柇璇樊鏄惁宸茬粡鍦?LSTM 杈撳嚭澶勭垎鎺夛紝鑰屼笉鏄户缁寽銆?"""


class LSTMLastStepDebug(nn.Module):
    """只返回 CNN-LSTM 最后一个时间步的 LSTM 输出。"""

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
        x = x[:, -1, :]
        return x


class LSTMInputDebug(nn.Module):
    """只返回进入 LSTM 之前的时序特征，用于和 dump 中对应张量对比。"""

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
        return x


class LSTMSequenceDebug(nn.Module):
    """返回 LSTM 全部时间步输出，用于比较 Y[0] 到 Y[4]。"""

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
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PTH vs kmodel on LSTM Y[-1] for one sample.")
    parser.add_argument("--infer_config", type=str, required=True)
    parser.add_argument("--export_config", type=str, required=True)
    parser.add_argument("--scheme_name", type=str, default="lstm_cutoff_debug")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--csv_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/layerwise/lstm_cutoff_debug")
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument(
        "--stage",
        type=str,
        default="lstm_last",
        choices=["lstm_last", "lstm_input", "lstm_sequence"],
        help="选择要截断对比的阶段：lstm_input 对应 LSTM 输入，lstm_last 对应 Y[-1]。",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_selected_sample(root: Path, infer_cfg: dict, data_dir: Path, sample_index: int, csv_name: Optional[str]):
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
    if X.shape[0] == 0:
        raise RuntimeError(f"娌℃湁浠庢暟鎹洰褰曟瀯寤哄嚭鏍锋湰: {data_dir}")

    scaler_path = (root / infer_cfg["normalization"]["scaler_path"]).resolve()
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    if csv_name:
        mask = source == csv_name
        if not np.any(mask):
            raise FileNotFoundError(f"鎸囧畾鐨?csv_name 涓嶅瓨鍦? {csv_name}")
        X_scaled = X_scaled[mask]
        y = y[mask]
        source = source[mask]

    if sample_index < 0 or sample_index >= int(X_scaled.shape[0]):
        raise IndexError(f"sample_index 瓒呭嚭鑼冨洿: {sample_index}, total={X_scaled.shape[0]}")

    return (
        X_scaled[sample_index : sample_index + 1],
        float(y[sample_index]),
        str(source[sample_index]),
        X_scaled,
        y,
    )


def build_debug_model(root: Path, infer_cfg: dict, input_shape):
    model_cfg = infer_cfg["model"]
    weights_path = (root / model_cfg["weights_path"]).resolve()
    state_dict = infer.load_state_dict_compat(weights_path, torch.device("cpu"))
    base_model = infer.build_model_from_config(model_cfg=model_cfg, input_shape=input_shape, state_dict=state_dict)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()
    return base_model


def run_pth_last_y(model: nn.Module, sample: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        value = model(torch.from_numpy(sample.astype(np.float32)))
    return value.detach().cpu().numpy().astype(np.float32)


def clone_export_cfg_for_cutoff(export_cfg: dict, root: Path, output_dir: Path, scheme_name: str):
    cloned = json.loads(json.dumps(export_cfg))
    cloned["name"] = f"{scheme_name}_cutoff"

    paths = cloned["paths"]
    paths["onnx"] = (output_dir / f"{scheme_name}_lstm_last.onnx").resolve().as_posix()
    paths["kmodel"] = (output_dir / f"{scheme_name}_lstm_last.kmodel").resolve().as_posix()
    paths["scaler_json"] = (output_dir / f"{scheme_name}_lstm_last_scaler.json").resolve().as_posix()
    paths["calibration_npy"] = (output_dir / f"{scheme_name}_lstm_last_calibration.npy").resolve().as_posix()
    paths["nncase_dump_dir"] = (output_dir / f"{scheme_name}_lstm_last_nncase_dump").resolve().as_posix()
    return cloned


def export_cutoff_onnx(model: nn.Module, onnx_path: Path, sample_shape):
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("导出截断 ONNX 需要安装 onnx 包。") from exc

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
        output_names=["lstm_last"],
        dynamic_axes=None,
    )
    build_kmodel.sanitize_onnx_for_nncase(onnx_path, onnx)


def run_kmodel_last_y(sample: np.ndarray, kmodel_path: Path) -> np.ndarray:
    compare_pth_kmodel.prepare_nncase_env()
    import nncase

    sim = nncase.Simulator()
    sim.load_model(kmodel_path.read_bytes())
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(sample.astype(np.float32)))
    sim.run()
    return np.asarray(sim.get_output_tensor(0).to_numpy(), dtype=np.float32)


def run_full_scalar_pair(sample: np.ndarray, infer_cfg: dict, export_cfg: dict, root: Path):
    pth_scalar = compare_pth_kmodel.run_pth_predictions(sample.astype(np.float32), infer_cfg, root)
    kmodel_path = Path(export_cfg["paths"]["kmodel"])
    if not kmodel_path.is_absolute():
        kmodel_path = (root / export_cfg["paths"]["kmodel"]).resolve()
    kmodel_scalar = compare_pth_kmodel.run_kmodel_predictions(sample.astype(np.float32), kmodel_path, log_every=1)
    return float(pth_scalar[0]), float(kmodel_scalar[0])


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    infer_cfg_path = Path(args.infer_config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = (root / infer_cfg_path).resolve()
    export_cfg_path = Path(args.export_config)
    if not export_cfg_path.is_absolute():
        export_cfg_path = (root / export_cfg_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = load_json(infer_cfg_path)
    export_cfg = load_json(export_cfg_path)

    sample, true_label, source_csv, X_scaled, y_all = build_selected_sample(
        root=root,
        infer_cfg=infer_cfg,
        data_dir=data_dir,
        sample_index=int(args.sample_index),
        csv_name=args.csv_name,
    )

    input_shape = tuple(sample.shape[1:])
    base_model = build_debug_model(root=root, infer_cfg=infer_cfg, input_shape=input_shape)
    if args.stage == "lstm_input":
        debug_model = LSTMInputDebug(base_model).eval()
        output_name = "lstm_input"
    elif args.stage == "lstm_sequence":
        debug_model = LSTMSequenceDebug(base_model).eval()
        output_name = "lstm_sequence"
    else:
        debug_model = LSTMLastStepDebug(base_model).eval()
        output_name = "lstm_last"

    pth_last = run_pth_last_y(debug_model, sample)

    cutoff_cfg = clone_export_cfg_for_cutoff(export_cfg, root=root, output_dir=output_dir, scheme_name=args.scheme_name)
    export_cutoff_onnx(debug_model, Path(cutoff_cfg["paths"]["onnx"]), sample_shape=input_shape)

    scaler_pkl = Path(export_cfg["paths"]["scaler_pkl"])
    if not scaler_pkl.is_absolute():
        scaler_pkl = (root / scaler_pkl).resolve()
    build_kmodel.export_scaler_json(scaler_pkl, Path(cutoff_cfg["paths"]["scaler_json"]))

    calib_count = int(args.max_calib_samples) if args.max_calib_samples is not None else build_kmodel.resolve_calibration_sample_count(
        export_cfg["quantization"].get("samples_count", 64),
        X_scaled.shape[0],
        "quantization.samples_count",
    )
    calibration_data = build_kmodel.select_calibration_data(
        X_scaled=X_scaled,
        count=calib_count,
        strategy=export_cfg["quantization"].get("sampling_strategy", "first"),
        random_seed=export_cfg["quantization"].get("random_seed", None),
        y_labels=y_all,
    )
    np.save(Path(cutoff_cfg["paths"]["calibration_npy"]), calibration_data.astype(np.float32))
    build_kmodel.compile_kmodel_with_nncase(cutoff_cfg, root=root, calibration_data=calibration_data.astype(np.float32))

    kmodel_last = run_kmodel_last_y(sample, Path(cutoff_cfg["paths"]["kmodel"]))

    diff = kmodel_last.reshape(-1) - pth_last.reshape(-1)
    pth_scalar, kmodel_scalar = run_full_scalar_pair(sample, infer_cfg=infer_cfg, export_cfg=export_cfg, root=root)

    top_abs_idx = np.argsort(np.abs(diff))[::-1][:10]
    top_abs_rows = []
    for idx in top_abs_idx.tolist():
        top_abs_rows.append(
            {
                "dim": int(idx),
                "pth": float(pth_last.reshape(-1)[idx]),
                "kmodel": float(kmodel_last.reshape(-1)[idx]),
                "abs_diff": float(abs(diff[idx])),
            }
        )

    per_timestep_rows = []
    if args.stage == "lstm_sequence":
        # 澶у皢鍐涳紝杩欓噷鍗曠嫭缁熻 5 涓椂闂存鐨勮宸紝渚夸簬鍒ゆ柇璇樊鏄惁闅忔椂闂寸疮绉斁澶с€?        for step_idx in range(int(pth_last.shape[1])):
            step_diff = kmodel_last[:, step_idx, :] - pth_last[:, step_idx, :]
            per_timestep_rows.append(
                {
                    "time_step": int(step_idx),
                    "mae": float(np.mean(np.abs(step_diff))),
                    "rmse": float(np.sqrt(np.mean(step_diff ** 2))),
                    "max_abs": float(np.max(np.abs(step_diff))),
                }
            )

    summary = {
        "sample_index": int(args.sample_index),
        "source_csv": source_csv,
        "true_label": true_label,
        "scheme_name": args.scheme_name,
        "stage": args.stage,
        "cutoff_kmodel_path": str(Path(cutoff_cfg["paths"]["kmodel"]).resolve()),
        "cutoff_onnx_path": str(Path(cutoff_cfg["paths"]["onnx"]).resolve()),
        f"pth_{output_name}_shape": list(pth_last.shape),
        f"kmodel_{output_name}_shape": list(kmodel_last.shape),
        f"{output_name}_mae": float(np.mean(np.abs(diff))),
        f"{output_name}_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{output_name}_max_abs": float(np.max(np.abs(diff))),
        "pth_scalar_output": float(pth_scalar),
        "kmodel_scalar_output": float(kmodel_scalar),
        "scalar_abs_diff": float(abs(kmodel_scalar - pth_scalar)),
        f"top10_{output_name}_abs_diff_dims": top_abs_rows,
    }
    if per_timestep_rows:
        summary["per_timestep_metrics"] = per_timestep_rows

    save_json(output_dir / f"{args.scheme_name}_summary.json", summary)
    np.savez(
        output_dir / f"{args.scheme_name}_tensors.npz",
        pth_output=pth_last.astype(np.float32),
        kmodel_output=kmodel_last.astype(np.float32),
        diff=diff.astype(np.float32),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
