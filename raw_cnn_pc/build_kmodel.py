import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Build K230 deploy assets for Raw+CNN.")
    parser.add_argument("--config", type=str, default="k230_export_config.json")
    # 大将军平时直接改配置文件即可；这里只保留一个临时覆盖入口，
    # 方便偶尔快速验证另一批校准数据，不影响默认配置用法。
    parser.add_argument(
        "--calibration_data_dir",
        type=str,
        default=None,
        help="Optional override for calibration data directory. If omitted, use paths.calibration_data_dir or paths.test_data_dir in config.",
    )
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


def build_dataset(
    data_dir: Path,
    base_window_size: int,
    base_step: int,
    seq_length: int,
    seq_step: int,
    feature_mode: str = "raw",
):
    # 这里把“目录里的原始长 CSV”转换成“模型真正吃的样本张量”。
    # 顺序是：
    # 1. 按文件名字典序遍历所有 CSV
    # 2. 逐个 CSV 按 base_window_size / base_step 切基础窗口
    # 3. 再按 sequence_length / sequence_step 组装成模型输入样本
    # 4. 最后把所有文件的样本顺序拼接成一个大数组
    X_list = []
    y_list = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = read_signal(csv_file)
        if signal.size < base_window_size:
            continue
        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            window = apply_feature_mode(window.astype(np.float32), feature_mode)
            features.append(window)
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


def normalize_feature_mode(feature_mode: str) -> str:
    text = str(feature_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"window_demean", "demean", "window_mean_center"}:
        return "window_demean"
    return "raw"


def apply_feature_mode(window: np.ndarray, feature_mode: str) -> np.ndarray:
    mode = normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        return (window - np.mean(window, dtype=np.float32)).astype(np.float32)
    return window.astype(np.float32)


def ensure_per_layer(value, num_layers: int, field: str):
    if isinstance(value, (list, tuple)):
        if len(value) != num_layers:
            raise ValueError(f"{field} length mismatch: {len(value)} vs {num_layers}")
        return list(value)
    return [value] * num_layers


class CNNAll(nn.Module):
    def __init__(self, input_shape, conv_filters, kernel_size=3, pool_size=2):
        super().__init__()
        time_steps, features = input_shape
        self.time_steps = int(time_steps)
        self.features = int(features)

        conv_filters = list(conv_filters)
        if not conv_filters:
            raise ValueError("conv_filters must not be empty.")
        num_layers = len(conv_filters)
        kernel_sizes = ensure_per_layer(kernel_size, num_layers, "kernel_size")
        pool_sizes = ensure_per_layer(pool_size, num_layers, "pool_size")

        in_channels = self.time_steps
        self.convs = nn.ModuleList()
        for out_channels, k in zip(conv_filters, kernel_sizes):
            k = int(k)
            self.convs.append(nn.Conv1d(in_channels, int(out_channels), kernel_size=k, padding=k // 2))
            in_channels = int(out_channels)
        self.pool_sizes = [int(p) for p in pool_sizes]
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(p) if p > 1 else nn.Identity() for p in self.pool_sizes]
        )

        length_after = self.features
        for p in self.pool_sizes:
            length_after = length_after // p
        flatten_dim = int(conv_filters[-1]) * int(length_after)
        self.fc = nn.Linear(flatten_dim, 1)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = torch.relu(conv(x))
            x = pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def export_onnx(model: nn.Module, onnx_path: Path, input_shape):
    try:
        import onnx  # type: ignore
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
    sanitize_onnx_for_nncase(onnx_path, onnx)


def sanitize_onnx_for_nncase(onnx_path: Path, onnx_module):
    # 某些 1D MaxPool 导出的 ONNX 会附带 dilations=[1] 属性，
    # 但当前 nncase 在导入时会把它当成尺寸错误的窗口参数。
    # 这里在导出后做一次轻量清洗，移除这项对结果无影响的属性。
    model = onnx_module.load(onnx_path.as_posix())
    changed = False
    for node in model.graph.node:
        if node.op_type != "MaxPool":
            continue
        keep_attrs = []
        for attr in node.attribute:
            if attr.name == "dilations":
                changed = True
                continue
            keep_attrs.append(attr)
        if len(keep_attrs) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(keep_attrs)
    if changed:
        onnx_module.save(model, onnx_path.as_posix())


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


def normalize_sampling_strategy(strategy: str) -> str:
    # 量化校准样本抽取策略统一在这里标准化，配置里写法可以更宽松。
    # first   : 直接取前 N 条样本
    # uniform : 从全体样本中均匀抽 N 条
    # random  : 从全体样本中随机抽 N 条
    text = str(strategy).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"first", "head", "sequential"}:
        return "first"
    if text in {"uniform", "even", "linspace"}:
        return "uniform"
    if text in {"random", "shuffle", "rand"}:
        return "random"
    raise ValueError(
        "quantization.sampling_strategy must be one of: first / uniform / random"
    )


def select_calibration_data(X_scaled: np.ndarray, count: int, strategy: str, random_seed):
    # 量化校准真正喂给 nncase 的样本在这里确定。
    # 大将军以后如果想调整“拿哪些样本去量化”，只需要改
    # k230_export_config.json 里的：
    # - quantization.samples_count
    # - quantization.sampling_strategy
    # - quantization.random_seed
    total = int(X_scaled.shape[0])
    if total <= 0:
        raise RuntimeError("No scaled samples available for calibration.")
    if count >= total:
        return X_scaled.astype(np.float32)

    mode = normalize_sampling_strategy(strategy)
    if mode == "first":
        # 保留旧逻辑：按当前样本顺序直接取前 N 条。
        indices = np.arange(count, dtype=np.int64)
    elif mode == "uniform":
        # 从整体样本范围内均匀取点，适合尽量覆盖全局分布。
        indices = np.linspace(0, total - 1, num=count, dtype=np.int64)
    else:
        # 随机抽样适合做对照实验；random_seed 固定时结果可复现。
        if random_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(int(random_seed))
        indices = np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))

    return X_scaled[indices].astype(np.float32)


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
    # 下面这些字段就是量化的核心配置：
    # samples_count      : 最终实际参与量化校准的样本条数
    # quant_type         : 激活值量化类型
    # weight_quant_type  : 权重量化类型
    # calibrate_method   : 量化范围估计方法
    ptq_options.samples_count = int(calibration_data.shape[0])
    ptq_options.quant_type = qcfg.get("quant_type", "uint8")
    ptq_options.w_quant_type = qcfg.get("weight_quant_type", "uint8")
    ptq_options.calibrate_method = qcfg.get("calibrate_method", "NoClip")

    # nncase 这里吃的是“样本列表”，每条样本 shape 都是 (1, C, L)。
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
    model_cfg = cfg["model"]
    qcfg = cfg["quantization"]
    feature_mode = normalize_feature_mode(cfg.get("preprocessing", {}).get("feature_mode", "raw"))

    weights_pth = (root / paths["weights_pth"]).resolve()
    onnx_path = (root / paths["onnx"]).resolve()
    scaler_pkl = (root / paths["scaler_pkl"]).resolve()
    scaler_json = (root / paths["scaler_json"]).resolve()
    calib_npy = (root / paths["calibration_npy"]).resolve()
    if args.calibration_data_dir:
        # 如果临时传了目录，就优先用临时目录。
        calibration_data_dir = Path(args.calibration_data_dir).resolve()
    else:
        # 正常长期使用时，大将军只需要改配置里的 calibration_data_dir。
        # 如果没写这个字段，才回退到旧字段 test_data_dir。
        calib_data_dir_cfg = paths.get("calibration_data_dir", paths["test_data_dir"])
        calibration_data_dir = (root / calib_data_dir_cfg).resolve()

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    try:
        X, y = build_dataset(
            data_dir=calibration_data_dir,
            base_window_size=base_window,
            base_step=base_step,
            seq_length=seq_length,
            seq_step=seq_step,
            feature_mode=feature_mode,
        )
        if X.shape[0] == 0:
            raise RuntimeError(f"No valid samples in calibration data: {calibration_data_dir}")

        X_scaled = apply_scaler(scaler_pkl, X)
        if args.max_calib_samples is not None:
            requested = require_positive_int(args.max_calib_samples, "max_calib_samples")
        else:
            # 平时优先改配置里的 quantization.samples_count；
            # 这里的 requested 只是“希望抽多少条”，真正能抽多少还要看总样本数。
            requested = require_positive_int(qcfg.get("samples_count", 64), "quantization.samples_count")
        count = min(requested, X_scaled.shape[0])
        sampling_strategy = qcfg.get("sampling_strategy", "first")
        random_seed = qcfg.get("random_seed", None)
        # 这里是“从全部候选样本里抽出量化校准子集”的唯一入口。
        calibration_data = select_calibration_data(
            X_scaled=X_scaled,
            count=count,
            strategy=sampling_strategy,
            random_seed=random_seed,
        )
        calib_npy.parent.mkdir(parents=True, exist_ok=True)
        # 额外保存一份 calibration_input.npy，方便后续排查到底用了哪批样本去量化。
        np.save(calib_npy, calibration_data)

        export_scaler_json(scaler_pkl, scaler_json)

        input_shape = tuple(X_scaled.shape[1:])
        model = CNNAll(
            input_shape=input_shape,
            conv_filters=model_cfg["conv_filters"],
            kernel_size=model_cfg["kernel_size"],
            pool_size=model_cfg["pool_size"],
        )
        state_dict = load_state_dict_compat(weights_pth, torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        export_onnx(model, onnx_path, input_shape=input_shape)

        print("Exported ONNX:", onnx_path)
        print("Exported scaler json:", scaler_json)
        print("Calibration data dir:", calibration_data_dir)
        print("Saved calibration data:", calib_npy, calibration_data.shape)
        print("Calibration sampling strategy:", normalize_sampling_strategy(sampling_strategy))
        print("Calibration random seed:", random_seed)
        print("feature_mode:", feature_mode)

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
