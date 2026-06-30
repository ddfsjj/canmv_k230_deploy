try:
    import sys
except ImportError:
    sys = None  # type: ignore

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore


def ensure_local_module_path():
    # 兼容从其他目录启动脚本，先把当前脚本目录加入模块搜索路径。
    if sys is None or not hasattr(sys, "path"):
        return

    script_dir = None
    try:
        argv = getattr(sys, "argv", None)
        if argv and len(argv) > 0 and argv[0]:
            entry = str(argv[0]).replace("\\", "/")
            if "/" in entry:
                script_dir = entry.rsplit("/", 1)[0]
    except Exception:
        script_dir = None

    try:
        here = globals().get("__file__", "")
        if (not script_dir) and here:
            norm_here = str(here).replace("\\", "/")
            if "/" in norm_here:
                script_dir = norm_here.rsplit("/", 1)[0]
    except Exception:
        pass

    if not script_dir:
        try:
            script_dir = os.getcwd()
        except Exception:
            script_dir = None

    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)


ensure_local_module_path()


def detect_script_dir():
    if sys is not None:
        try:
            argv = getattr(sys, "argv", None)
            if argv and len(argv) > 0 and argv[0]:
                entry = str(argv[0]).replace("\\", "/")
                if "/" in entry:
                    return entry.rsplit("/", 1)[0]
        except Exception:
            pass

    try:
        here = globals().get("__file__", "")
        if here:
            norm_here = str(here).replace("\\", "/")
            if "/" in norm_here:
                return norm_here.rsplit("/", 1)[0]
    except Exception:
        pass

    try:
        return os.getcwd()
    except Exception:
        return None


def load_base_module_from_file():
    # 某些 CanMV IDE 直接运行脚本时，同目录模块不一定能正常 import，这里做兜底加载。
    script_dir = detect_script_dir()
    candidates = []
    if script_dir:
        candidates.append(str(script_dir).replace("\\", "/") + "/legacy/run_k230_infer_legacy.py")
    candidates.append("/sdcard/raw_cnn_k230/legacy/run_k230_infer_legacy.py")
    candidates.append("legacy/run_k230_infer_legacy.py")

    last_error = None
    for path in candidates:
        try:
            with open(path, "r") as f:
                source = f.read()
            module_globals = {
                "__name__": "run_k230_infer_legacy",
                "__file__": path,
            }
            exec(source, module_globals)

            class ModuleProxy:
                pass

            proxy = ModuleProxy()
            for key, value in module_globals.items():
                setattr(proxy, key, value)
            return proxy
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise ImportError("Cannot load legacy/run_k230_infer_legacy.py")


try:
    from legacy import run_k230_infer_legacy as base
except ImportError:
    base = load_base_module_from_file()


OUTPUT_CSV_NAME = "predictions_k230_compare.csv"
PER_CSV_OUTPUT_NAME = "predictions_k230_per_csv.csv"


def parse_compare_samples(cli_args, runtime_cfg):
    # 优先级：
    # 1. 命令行参数：python run_k230_csv_compare.py 50 / all
    # 2. k230_config.json 里的 runtime.csv_cached.compare_max_samples
    # 3. 默认为 10
    if cli_args:
        token = str(cli_args[0]).strip().lower()
        if token in {"all", "full", "*"}:
            return None
        value = int(token)
        if value <= 0:
            raise ValueError("compare sample count must be > 0, or use `all`.")
        return value

    csv_cfg = base.get_runtime_section(runtime_cfg, "csv_cached")
    cfg_value = csv_cfg.get("compare_max_samples", runtime_cfg.get("compare_max_samples", 10))
    if cfg_value is None:
        return None

    value = int(cfg_value)
    if value <= 0:
        raise ValueError("runtime.csv_cached.compare_max_samples must be > 0, or null for all.")
    return value


def build_dataset_for_single_csv(cfg, csv_path, max_samples=None):
    # 单文件构建样本，避免整批 CSV 一次性堆满内存。
    data_cfg = cfg["data"]
    base_window = base.require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = base.resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = base.require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = base.require_positive_int(data_cfg["sequence_step"], "data.sequence_step")
    feature_mode = base.get_feature_mode(cfg)
    if max_samples is not None:
        max_samples = base.require_positive_int(max_samples, "compare.remaining_samples")

    signal = base.read_signal(csv_path)
    if signal.size < base_window:
        return base.empty_float((0, seq_length, 0)), base.empty_float((0,))

    label = base.parse_label_from_name(csv_path)
    X_list = []
    y_list = []
    features = []
    next_emit_start = 0

    for start in range(0, signal.size - base_window + 1, base_step):
        window = base.astype_float_array(signal[start : start + base_window])
        proc_window = base.empty_float((base_window,))
        base.apply_feature_mode_1d(window, feature_mode, proc_window)
        features.append(proc_window)

        while next_emit_start + seq_length <= len(features):
            sample = base.empty_float((seq_length, base_window))
            seg = features[next_emit_start : next_emit_start + seq_length]
            for j in range(seq_length):
                sample[j] = seg[j]
            X_list.append(sample)
            y_list.append(label)
            next_emit_start += seq_step

            if max_samples is not None and len(X_list) >= max_samples:
                return base.finalize_dataset(X_list, y_list, seq_length)

    return base.finalize_dataset(X_list, y_list, seq_length)


def init_prediction_csv(path):
    # 逐样本结果追加 csv_name，便于定位某个预测来自哪个干度文件。
    base.ensure_dir(base.dirname(path))
    with open(path, "w") as f:
        f.write("sample_id,csv_name,true_label,prediction\n")


def append_prediction_rows(path, start_sample_id, csv_name, y_true, y_pred):
    with open(path, "a") as f:
        for i in range(len(y_pred)):
            f.write(
                "{},{},{},{}\n".format(
                    int(start_sample_id + i),
                    csv_name,
                    float(y_true[i]),
                    float(y_pred[i]),
                )
            )


def init_per_csv_summary(path):
    # 每个文件一行汇总，便于直接按干度看整体效果。
    base.ensure_dir(base.dirname(path))
    with open(path, "w") as f:
        f.write("csv_name,true_label,sample_count,pred_mean,mae,rmse\n")


def append_per_csv_summary(path, csv_name, true_label, sample_count, y_true, y_pred):
    pred_sum = 0.0
    for v in y_pred:
        pred_sum += float(v)
    pred_mean = pred_sum / float(sample_count)
    mae = base.safe_metric_mae(y_true, y_pred)
    rmse = base.safe_metric_rmse(y_true, y_pred)
    with open(path, "a") as f:
        f.write(
            "{},{},{},{},{},{}\n".format(
                csv_name,
                float(true_label),
                int(sample_count),
                pred_mean,
                mae,
                rmse,
            )
        )


def main():
    # 改成逐 CSV 流式预测，避免板端在全量比较时因内存不足卡死。
    cli_args = []
    if sys is not None:
        try:
            cli_args = list(sys.argv[1:])
        except Exception:
            cli_args = []
    root = base.detect_root()
    config_path = base.resolve_runtime_config_path(root, cli_args)
    cfg = base.load_json(config_path)

    runtime_cfg = cfg.get("runtime", {})
    compare_samples = parse_compare_samples(cli_args, runtime_cfg)

    runtime_cfg["mode"] = "csv_cached"
    csv_cfg = base.get_runtime_section(runtime_cfg, "csv_cached")
    csv_cfg["max_samples"] = compare_samples
    csv_cfg["write_predictions_csv"] = True
    runtime_cfg["csv_cached"] = csv_cfg
    cfg["runtime"] = runtime_cfg

    uart_cfg = cfg.get("uart", {})
    uart_cfg["enabled"] = False
    cfg["uart"] = uart_cfg

    paths = cfg["paths"]
    paths["predictions_csv"] = OUTPUT_CSV_NAME
    cfg["paths"] = paths

    scaler_json_path = base.join_path(root, paths["scaler_json"])
    kmodel_path = base.join_path(root, paths["kmodel"])
    data_dir = base.join_path(root, paths["test_data_dir"])
    pred_csv = base.join_path(root, OUTPUT_CSV_NAME)
    per_csv_summary = base.join_path(root, PER_CSV_OUTPUT_NAME)
    pred_csv = base.path_with_kmodel_name(pred_csv, kmodel_path)
    per_csv_summary = base.path_with_kmodel_name(per_csv_summary, kmodel_path)

    csv_files = base.list_csv_files(data_dir)
    if not csv_files:
        raise RuntimeError("No csv files found in test_data_dir: " + str(data_dir))

    init_prediction_csv(pred_csv)
    init_per_csv_summary(per_csv_summary)

    t_start = base.now_us()
    total_samples = 0
    processed_csv = 0
    infer_us_total = 0
    mae_weighted_sum = 0.0
    rmse_square_weighted_sum = 0.0
    remaining = compare_samples
    sample_id_base = 0
    first_input_shape = None
    first_10_predictions = []
    model_reloaded_any = False

    for csv_path in csv_files:
        if remaining is not None and remaining <= 0:
            break

        print("dataset_read_csv:", csv_path)
        X, y = build_dataset_for_single_csv(cfg, csv_path, max_samples=remaining)
        sample_count = int(X.shape[0])
        csv_name = base.norm_path(csv_path).split("/")[-1]
        if sample_count <= 0:
            print("dataset_skip_csv_no_samples:", csv_name)
            continue

        if first_input_shape is None:
            first_input_shape = tuple(X.shape[1:])

        X_scaled = base.scale_features(X, scaler_json_path)
        del X
        base.gc.collect()

        y_pred, infer_us_one, model_reloaded = base.run_kmodel_inference_cached(
            kmodel_path=kmodel_path,
            X_scaled=X_scaled,
            start_idx=0,
            count=sample_count,
            uart_sender=None,
        )
        del X_scaled
        base.gc.collect()

        append_prediction_rows(pred_csv, sample_id_base, csv_name, y, y_pred)
        append_per_csv_summary(
            per_csv_summary,
            csv_name,
            base.parse_label_from_name(csv_name),
            sample_count,
            y,
            y_pred,
        )

        csv_mae = base.safe_metric_mae(y, y_pred)
        csv_rmse = base.safe_metric_rmse(y, y_pred)
        mae_weighted_sum += float(csv_mae) * float(sample_count)
        rmse_square_weighted_sum += float(csv_rmse) * float(csv_rmse) * float(sample_count)
        infer_us_total += infer_us_one
        total_samples += sample_count
        processed_csv += 1
        sample_id_base += sample_count
        model_reloaded_any = bool(model_reloaded_any or model_reloaded)

        if len(first_10_predictions) < 10:
            need = 10 - len(first_10_predictions)
            first_10_predictions.extend(y_pred[:need].tolist())

        if remaining is not None:
            remaining -= sample_count

        print(
            "csv_done: csv_name={}, samples={}, total_samples={}, processed_csv={}".format(
                csv_name,
                sample_count,
                total_samples,
                processed_csv,
            )
        )

        del y
        del y_pred
        base.gc.collect()

    if total_samples <= 0:
        raise RuntimeError("No valid samples available for comparison.")

    t_end = base.now_us()
    total_us = base.diff_us(t_end, t_start)
    mae = mae_weighted_sum / float(total_samples)
    rmse = (rmse_square_weighted_sum / float(total_samples)) ** 0.5

    print("=== K230 Raw+CNN Compare ===")
    print("root:", root)
    print("config_path:", config_path)
    print("mode:", "csv_cached")
    print("kmodel:", kmodel_path)
    print("dataset_total_samples:", total_samples)
    print("processed_csv_files:", processed_csv)
    print("infer_batch_size:", "per_csv_stream")
    print("infer_start_idx:", 0)
    print("model_reloaded:", bool(model_reloaded_any))
    print("input_shape:", first_input_shape)
    print("model_infer_time_sec:", infer_us_total / 1_000_000.0)
    print("model_infer_time_per_sample_ms:", infer_us_total / 1000.0 / float(total_samples))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("prediction_csv:", pred_csv)
    print("per_csv_summary:", per_csv_summary)
    print("first_10_predictions:", first_10_predictions)
    if compare_samples is None:
        print("compare_samples:", "all")
    else:
        print("compare_samples:", int(compare_samples))


if __name__ == "__main__":
    main()
