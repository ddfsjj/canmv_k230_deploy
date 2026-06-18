import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import compare_pth_kmodel
import build_kmodel

"""
閲忓寲鏂规鎵归噺璇勪及鑴氭湰銆?
浣滅敤锛?1. 棰勮澶氬閲忓寲鏂规銆?2. 涓烘瘡濂楁柟妗堝崟鐙鍑?`onnx/kmodel` 鍜屾牎鍑嗕骇鐗┿€?3. 瀵规瘡濂楁柟妗堣窇鍏ㄩ噺 `.pth vs kmodel` 瀵规瘮銆?4. 鏈€缁堢敓鎴愪竴浠?Markdown 鎶ュ憡锛屾柟渚跨洿鎺ョ湅缁撹銆?"""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple K230 quantization schemes on full dataset.")
    parser.add_argument("--infer_config", type=str, default="infer_config.json")
    parser.add_argument("--export_config", type=str, default="k230_export_config.json")
    parser.add_argument(
        "--schemes_json",
        type=str,
        default=None,
        help="Optional custom scheme list json path. If omitted, use built-in scheme list.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/code/network/VQ_Estimator/data/generated_dry_temp_csv",
        help="Full evaluation data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scheme_eval",
        help="Directory under raw_cnn_pc to save per-scheme outputs and final markdown report.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="Only run one scheme id. If omitted, only report generation runs over existing results.",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Skip evaluation and only regenerate markdown report from existing outputs.",
    )
    parser.add_argument("--log_every", type=int, default=2000)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def get_scheme_list():
    # 这里保留已经确认过、且当前工具链允许的候选量化方案。
    return [
        {
            "id": "u8_u8_kld_256",
            "title": "uint8 + uint8 + Kld + 256",
            "samples_count": 256,
            "sampling_strategy": "first",
            "quant_type": "uint8",
            "weight_quant_type": "uint8",
            "calibrate_method": "Kld",
        },
        {
            "id": "u8_i16_kld_256",
            "title": "uint8 + int16 + Kld + 256",
            "samples_count": 256,
            "sampling_strategy": "first",
            "quant_type": "uint8",
            "weight_quant_type": "int16",
            "calibrate_method": "Kld",
        },
        {
            "id": "i16_u8_kld_256",
            "title": "int16 + uint8 + Kld + 256",
            "samples_count": 256,
            "sampling_strategy": "first",
            "quant_type": "int16",
            "weight_quant_type": "uint8",
            "calibrate_method": "Kld",
        },
        {
            "id": "u8_i16_noclip_256",
            "title": "uint8 + int16 + NoClip + 256",
            "samples_count": 256,
            "sampling_strategy": "first",
            "quant_type": "uint8",
            "weight_quant_type": "int16",
            "calibrate_method": "NoClip",
        },
        {
            "id": "u8_i16_kld_512",
            "title": "uint8 + int16 + Kld + 512",
            "samples_count": 512,
            "sampling_strategy": "first",
            "quant_type": "uint8",
            "weight_quant_type": "int16",
            "calibrate_method": "Kld",
        },
    ]


def load_scheme_list_from_json(path: Path):
    # 支持把方案列表放到外部 json，方便针对某一版模型单独做量化筛选。
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid schemes json, expected non-empty list: {path}")
    required_fields = {
        "id",
        "title",
        "samples_count",
        "sampling_strategy",
        "quant_type",
        "weight_quant_type",
        "calibrate_method",
    }
    out = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid scheme at index {idx}: expected object")
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise ValueError(
                "Scheme {} missing fields: {}".format(item.get("id", idx), ", ".join(missing))
            )
        out.append(item)
    return out


def find_scheme(scheme_id: str, schemes=None):
    if schemes is None:
        schemes = get_scheme_list()
    for scheme in schemes:
        if scheme["id"] == scheme_id:
            return scheme
    raise KeyError(f"Unknown scheme id: {scheme_id}")


def make_scheme_export_cfg(base_cfg: dict, scheme: dict):
    cfg = deepcopy(base_cfg)
    qcfg = cfg["quantization"]
    qcfg["samples_count"] = scheme["samples_count"]
    qcfg["sampling_strategy"] = scheme["sampling_strategy"]
    qcfg["quant_type"] = scheme["quant_type"]
    qcfg["weight_quant_type"] = scheme["weight_quant_type"]
    qcfg["calibrate_method"] = scheme["calibrate_method"]

    # 为每个方案生成独立产物，避免同名文件互相覆盖。
    cfg["paths"]["onnx"] = f"scheme_eval/models/{scheme['id']}.onnx"
    cfg["paths"]["kmodel"] = f"scheme_eval/models/{scheme['id']}.kmodel"
    cfg["paths"]["scaler_json"] = f"scheme_eval/models/{scheme['id']}_scaler.json"
    cfg["paths"]["calibration_npy"] = f"scheme_eval/models/{scheme['id']}_calibration.npy"
    cfg["paths"]["nncase_dump_dir"] = f"scheme_eval/models/{scheme['id']}_nncase_dump"
    return cfg


def build_scheme_model(root: Path, scheme_cfg: dict):
    paths = scheme_cfg["paths"]
    data_cfg = scheme_cfg["data"]
    model_cfg = scheme_cfg["model"]
    qcfg = scheme_cfg["quantization"]
    feature_mode = build_kmodel.normalize_feature_mode(
        scheme_cfg.get("preprocessing", {}).get("feature_mode", "raw")
    )

    weights_pth = (root / paths["weights_pth"]).resolve()
    onnx_path = (root / paths["onnx"]).resolve()
    scaler_pkl = (root / paths["scaler_pkl"]).resolve()
    scaler_json = (root / paths["scaler_json"]).resolve()
    calib_npy = (root / paths["calibration_npy"]).resolve()
    calibration_data_dir = (root / paths["calibration_data_dir"]).resolve()

    base_window = build_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step = build_kmodel.resolve_positive_step(
        data_cfg.get("base_step", None),
        base_window // 2,
        "data.base_step",
    )
    seq_length = build_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = build_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    X, y = build_kmodel.build_dataset(
        data_dir=calibration_data_dir,
        base_window_size=base_window,
        base_step=base_step,
        seq_length=seq_length,
        seq_step=seq_step,
        feature_mode=feature_mode,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"No valid calibration samples in: {calibration_data_dir}")

    X_scaled = build_kmodel.apply_scaler(scaler_pkl, X)
    count = build_kmodel.resolve_calibration_sample_count(
        qcfg.get("samples_count", 64),
        X_scaled.shape[0],
        "quantization.samples_count",
    )
    calibration_data = build_kmodel.select_calibration_data(
        X_scaled=X_scaled,
        count=count,
        strategy=qcfg.get("sampling_strategy", "first"),
        random_seed=qcfg.get("random_seed", None),
        y_labels=y,
    )

    calib_npy.parent.mkdir(parents=True, exist_ok=True)
    build_kmodel.export_scaler_json(scaler_pkl, scaler_json)
    build_kmodel.np.save(calib_npy, calibration_data)

    input_shape = tuple(X_scaled.shape[1:])
    state_dict = build_kmodel.load_state_dict_compat(weights_pth, build_kmodel.torch.device("cpu"))
    model = build_kmodel.build_model_from_config(
        model_cfg=model_cfg,
        input_shape=input_shape,
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    build_kmodel.export_onnx(model, onnx_path, input_shape=input_shape)
    build_kmodel.compile_kmodel_with_nncase(scheme_cfg, root, calibration_data)

    return {
        "calibration_dir": str(calibration_data_dir),
        "calibration_total_candidates": int(X_scaled.shape[0]),
        "calibration_used_samples": int(calibration_data.shape[0]),
        "onnx_path": str(onnx_path),
        "kmodel_path": str((root / paths["kmodel"]).resolve()),
        "scaler_json": str(scaler_json),
        "calibration_npy": str(calib_npy),
    }


def run_scheme_compare(
    root: Path,
    infer_cfg_path: Path,
    export_cfg_path: Path,
    data_dir: Path,
    output_dir: Path,
    scheme: dict,
    max_samples=None,
    max_per_dryness=None,
    log_every: int = 2000,
):
    args = argparse.Namespace(
        infer_config=str(infer_cfg_path),
        export_config=str(export_cfg_path),
        data_dir=str(data_dir),
        max_samples=max_samples,
        max_per_dryness=max_per_dryness,
        summary_json=str(output_dir / f"{scheme['id']}_summary.json"),
        details_csv=str(output_dir / f"{scheme['id']}_details.csv"),
        per_csv_csv=str(output_dir / f"{scheme['id']}_per_csv.csv"),
        per_dryness_csv=str(output_dir / f"{scheme['id']}_per_dryness.csv"),
        log_every=log_every,
    )

    infer_cfg = compare_pth_kmodel.load_json(Path(args.infer_config))
    export_cfg = compare_pth_kmodel.load_json(Path(args.export_config))
    feature_mode = compare_pth_kmodel.infer.normalize_feature_mode(
        infer_cfg.get("preprocessing", {}).get("feature_mode", "raw")
    )
    data_cfg = infer_cfg["data"]

    X, y, source = compare_pth_kmodel.build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=compare_pth_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=compare_pth_kmodel.require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=compare_pth_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=compare_pth_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=feature_mode,
        max_samples=None,
    )
    raw_total_samples = int(X.shape[0])
    X, y, source = compare_pth_kmodel.apply_sample_limits(
        X,
        y,
        source,
        max_samples=args.max_samples,
        max_per_dryness=args.max_per_dryness,
    )
    if X.shape[0] == 0:
        raise RuntimeError("No samples left after applying sample limits.")
    scaler_path = root / infer_cfg["normalization"]["scaler_path"]
    scaler_path = scaler_path.resolve()
    scaler = compare_pth_kmodel.joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(compare_pth_kmodel.np.float32)

    t_pth_start = compare_pth_kmodel.time.perf_counter()
    pth_pred = compare_pth_kmodel.run_pth_predictions(X_scaled, infer_cfg, root)
    t_pth_end = compare_pth_kmodel.time.perf_counter()

    kmodel_path = root / export_cfg["paths"]["kmodel"]
    kmodel_path = kmodel_path.resolve()
    t_k_start = compare_pth_kmodel.time.perf_counter()
    kmodel_pred = compare_pth_kmodel.run_kmodel_predictions(X_scaled, kmodel_path, args.log_every)
    t_k_end = compare_pth_kmodel.time.perf_counter()

    summary = compare_pth_kmodel.make_summary(y, pth_pred, kmodel_pred)
    summary["data_dir"] = str(data_dir)
    summary["csv_file_count"] = len(sorted(data_dir.glob("*.csv")))
    summary["feature_mode"] = feature_mode
    summary["raw_total_samples_before_limit"] = raw_total_samples
    summary["max_samples"] = args.max_samples
    summary["max_per_dryness"] = args.max_per_dryness
    summary["pth_infer_time_sec"] = float(t_pth_end - t_pth_start)
    summary["kmodel_infer_time_sec"] = float(t_k_end - t_k_start)
    summary["pipeline_total_time_sec"] = float(compare_pth_kmodel.time.perf_counter() - t_pth_start)

    per_csv_rows = compare_pth_kmodel.make_per_csv_rows(source, y, pth_pred, kmodel_pred)
    per_dryness_rows = compare_pth_kmodel.make_per_dryness_rows(y, pth_pred, kmodel_pred)
    detail_rows = compare_pth_kmodel.make_detail_rows(source, y, pth_pred, kmodel_pred)

    compare_pth_kmodel.save_json(Path(args.summary_json), summary)
    compare_pth_kmodel.save_rows_csv(
        Path(args.per_csv_csv),
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
    compare_pth_kmodel.save_rows_csv(
        Path(args.per_dryness_csv),
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
    compare_pth_kmodel.save_rows_csv(
        Path(args.details_csv),
        ["sample_id", "csv_name", "true_label", "pth_prediction", "kmodel_prediction", "abs_diff"],
        detail_rows,
    )


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_markdown_report(output_dir: Path):
    schemes = get_scheme_list()
    available = []
    for scheme in schemes:
        summary_path = output_dir / f"{scheme['id']}_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        per_dryness = read_csv_rows(output_dir / f"{scheme['id']}_per_dryness.csv")
        details = read_csv_rows(output_dir / f"{scheme['id']}_details.csv")
        available.append((scheme, summary, per_dryness, details))

    lines = []
    lines.append("# KModel 量化方案全量测评报告")
    lines.append("")
    lines.append("本报告使用同一份原始数据目录做全量对比：")
    lines.append("")
    lines.append("- `D:/code/network/VQ_Estimator/data/generated_dry_temp_csv`")
    lines.append("")
    lines.append("对每套方案都执行：")
    lines.append("")
    lines.append("1. 导出独立 `kmodel`")
    lines.append("2. 用全量样本跑 `pth` 与 `kmodel` 对比")
    lines.append("3. 统计总体、按干度、按 CSV 的误差")
    lines.append("")

    if not available:
        lines.append("当前还没有任何方案结果文件。")
        return "\n".join(lines)

    ranked = sorted(available, key=lambda item: item[1]["kmodel_mae_vs_true"])
    best_by_mae = ranked[0]
    best_by_pth_gap = sorted(available, key=lambda item: item[1]["pth_vs_kmodel_mae"])[0]
    lines.append("## 最终建议")
    lines.append("")
    lines.append(
        "- 按 `kmodel_mae_vs_true` 看，当前最优方案是：`{}`".format(best_by_mae[0]["title"])
    )
    lines.append(
        "- 按 `pth_vs_kmodel_mae` 看，当前最接近 `pth` 的方案是：`{}`".format(best_by_pth_gap[0]["title"])
    )
    lines.append(
        "- 当前不建议保留全量校准版：`uint8 + int16 + Kld + full`，因为它在整体 `MAE/RMSE` 和 `P95/P99` 上都明显更差。"
    )
    lines.append(
        "- 报告里的最大绝对差会被少数离群样本放大，所以判断主方案时应优先看：`kmodel_mae_vs_true`、`kmodel_rmse_vs_true`、`pth_vs_kmodel_mae`、`P95/P99`。"
    )
    lines.append("")
    lines.append("## 总体排名")
    lines.append("")
    lines.append("| 排名 | 方案 | PTH MAE | KMODEL MAE | PTH RMSE | KMODEL RMSE | PTH vs K MAE | PTH vs K RMSE | 最大绝对差 | P95 | P99 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, (scheme, summary, _, _) in enumerate(ranked, start=1):
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                idx,
                scheme["title"],
                summary["pth_mae_vs_true"],
                summary["kmodel_mae_vs_true"],
                summary["pth_rmse_vs_true"],
                summary["kmodel_rmse_vs_true"],
                summary["pth_vs_kmodel_mae"],
                summary["pth_vs_kmodel_rmse"],
                summary["pth_vs_kmodel_max_abs"],
                summary["pth_vs_kmodel_p95_abs"],
                summary["pth_vs_kmodel_p99_abs"],
            )
        )
    lines.append("")

    for scheme, summary, per_dryness, details in available:
        better = []
        worse = []
        for row in per_dryness:
            pth_mae = float(row["pth_mae_vs_true"])
            k_mae = float(row["kmodel_mae_vs_true"])
            gap = k_mae - pth_mae
            item = {
                "dryness": float(row["dryness_label"]),
                "samples": int(float(row["samples"])),
                "pth_mae": pth_mae,
                "k_mae": k_mae,
                "pth_rmse": float(row["pth_rmse_vs_true"]),
                "k_rmse": float(row["kmodel_rmse_vs_true"]),
                "pk_mae": float(row["pth_vs_kmodel_mae"]),
                "pk_rmse": float(row["pth_vs_kmodel_rmse"]),
                "pk_max": float(row["pth_vs_kmodel_max_abs"]),
                "gap": gap,
            }
            if gap < 0:
                better.append(item)
            else:
                worse.append(item)

        large_outliers = [row for row in details if float(row["abs_diff"]) > 0.5]
        top_worse = sorted(worse, key=lambda x: x["gap"], reverse=True)[:5]
        top_better = sorted(better, key=lambda x: x["gap"])[:5]

        lines.append(f"## {scheme['title']}")
        lines.append("")
        lines.append("### 配置")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "samples_count": scheme["samples_count"],
                    "sampling_strategy": scheme["sampling_strategy"],
                    "quant_type": scheme["quant_type"],
                    "weight_quant_type": scheme["weight_quant_type"],
                    "calibrate_method": scheme["calibrate_method"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("### 总体结果")
        lines.append("")
        lines.append("- `total_samples = {}`".format(summary["total_samples"]))
        lines.append("- `pth_mae_vs_true = {:.6f}`".format(summary["pth_mae_vs_true"]))
        lines.append("- `pth_rmse_vs_true = {:.6f}`".format(summary["pth_rmse_vs_true"]))
        lines.append("- `kmodel_mae_vs_true = {:.6f}`".format(summary["kmodel_mae_vs_true"]))
        lines.append("- `kmodel_rmse_vs_true = {:.6f}`".format(summary["kmodel_rmse_vs_true"]))
        lines.append("- `pth_vs_kmodel_mae = {:.6f}`".format(summary["pth_vs_kmodel_mae"]))
        lines.append("- `pth_vs_kmodel_rmse = {:.6f}`".format(summary["pth_vs_kmodel_rmse"]))
        lines.append("- `pth_vs_kmodel_max_abs = {:.6f}`".format(summary["pth_vs_kmodel_max_abs"]))
        lines.append("- `pth_vs_kmodel_p95_abs = {:.6f}`".format(summary["pth_vs_kmodel_p95_abs"]))
        lines.append("- `pth_vs_kmodel_p99_abs = {:.6f}`".format(summary["pth_vs_kmodel_p99_abs"]))
        lines.append("- `kmodel_infer_time_sec = {:.3f}`".format(summary["kmodel_infer_time_sec"]))
        lines.append("")
        lines.append("### 干度层面结论")
        lines.append("")
        lines.append("- `kmodel` 比 `pth` 更好的干度数：`{}`".format(len(better)))
        lines.append("- `kmodel` 比 `pth` 更差的干度数：`{}`".format(len(worse)))
        lines.append("- `abs_diff > 0.5` 的离群样本数：`{}`".format(len(large_outliers)))
        lines.append("")
        lines.append("#### 变差最明显的 5 个干度")
        lines.append("")
        for item in top_worse:
            lines.append(
                "- 干度 `{:.12f}`：`pth_mae={:.6f}`，`kmodel_mae={:.6f}`，`pth_vs_k_mae={:.6f}`，`pth_vs_k_max={:.6f}`".format(
                    item["dryness"], item["pth_mae"], item["k_mae"], item["pk_mae"], item["pk_max"]
                )
            )
        lines.append("")
        lines.append("#### 改善最明显的 5 个干度")
        lines.append("")
        for item in top_better:
            lines.append(
                "- 干度 `{:.12f}`：`pth_mae={:.6f}`，`kmodel_mae={:.6f}`，`pth_vs_k_mae={:.6f}`，`pth_vs_k_max={:.6f}`".format(
                    item["dryness"], item["pth_mae"], item["k_mae"], item["pk_mae"], item["pk_max"]
                )
            )
        lines.append("")
        if large_outliers:
            lines.append("#### 离群样本")
            lines.append("")
            for row in large_outliers[:10]:
                lines.append(
                    "- `sample_id={}`，`csv={}`，`true={}`，`pth={}`，`kmodel={}`，`abs_diff={}`".format(
                        row["sample_id"],
                        row["csv_name"],
                        row["true_label"],
                        row["pth_prediction"],
                        row["kmodel_prediction"],
                        row["abs_diff"],
                    )
                )
            lines.append("")

        lines.append("### 结果文件")
        lines.append("")
        lines.append(f"- 摘要：`scheme_eval/{scheme['id']}_summary.json`")
        lines.append(f"- 按干度：`scheme_eval/{scheme['id']}_per_dryness.csv`")
        lines.append(f"- 按 CSV：`scheme_eval/{scheme['id']}_per_csv.csv`")
        lines.append(f"- 样本明细：`scheme_eval/{scheme['id']}_details.csv`")
        lines.append("")

    return "\n".join(lines)


def main():
    # 这里把“执行单个方案”和“仅重新生成报告”拆开，方便分批跑方案，最后统一汇总。
    args = parse_args()
    root = Path(__file__).resolve().parent
    infer_cfg_path = (root / args.infer_config).resolve() if not Path(args.infer_config).is_absolute() else Path(args.infer_config)
    export_cfg_path = (root / args.export_config).resolve() if not Path(args.export_config).is_absolute() else Path(args.export_config)
    data_dir = Path(args.data_dir).resolve()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only and args.scheme:
        scheme = find_scheme(args.scheme)
        base_export_cfg = load_json(export_cfg_path)
        scheme_export_cfg = make_scheme_export_cfg(base_export_cfg, scheme)
        scheme_cfg_path = output_dir / f"{scheme['id']}_export_config.json"
        save_json(scheme_cfg_path, scheme_export_cfg)

        print("=== Build Scheme ===")
        print("scheme:", scheme["id"])
        build_info = build_scheme_model(root, scheme_export_cfg)
        save_json(output_dir / f"{scheme['id']}_build_info.json", build_info)

        print("=== Compare Scheme ===")
        print("scheme:", scheme["id"])
        run_scheme_compare(root, infer_cfg_path, scheme_cfg_path, data_dir, output_dir, scheme)

    report_text = build_markdown_report(output_dir)
    report_path = output_dir / "quantization_scheme_report.md"
    save_text(report_path, report_text)
    print("report_path:", report_path)


if __name__ == "__main__":
    main()
