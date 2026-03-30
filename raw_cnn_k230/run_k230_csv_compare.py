try:
    import sys
except ImportError:
    sys = None  # type: ignore

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore


def ensure_local_module_path():
    # 兼容从其他目录启动脚本的场景，先把当前脚本所在目录加入模块搜索路径。
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
    # 某些 CanMV IDE 运行当前脚本时，不会把同目录自动加入 import 搜索路径。
    # 这里在常规 import 失败后，退化为直接读取同目录脚本并 exec。
    script_dir = detect_script_dir()
    candidates = []
    if script_dir:
        candidates.append(str(script_dir).replace("\\", "/") + "/run_k230_infer.py")
    candidates.append("/sdcard/raw_cnn_k230/run_k230_infer.py")
    candidates.append("run_k230_infer.py")

    last_error = None
    for path in candidates:
        try:
            with open(path, "r") as f:
                source = f.read()
            module_globals = {
                "__name__": "run_k230_infer",
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
    raise ImportError("Cannot load run_k230_infer.py")


try:
    import run_k230_infer as base
except ImportError:
    base = load_base_module_from_file()


# 默认输出文件名，单独和在线推理解耦，避免覆盖其他流程产物。
OUTPUT_CSV_NAME = "predictions_k230_compare.csv"


def parse_compare_samples(cli_args, runtime_cfg):
    # 优先级：
    # 1. 命令行参数：python run_k230_csv_compare.py 50 / all
    # 2. k230_config.json 里的 runtime.compare_max_samples
    # 3. 兜底默认值 10
    if cli_args:
        token = str(cli_args[0]).strip().lower()
        if token in {"all", "full", "*"}:
            return None
        value = int(token)
        if value <= 0:
            raise ValueError("compare sample count must be > 0, or use `all`.")
        return value

    cfg_value = runtime_cfg.get("compare_max_samples", 10)
    if cfg_value is None:
        return None

    value = int(cfg_value)
    if value <= 0:
        raise ValueError("runtime.compare_max_samples must be > 0, or null for all.")
    return value


def main():
    root = base.detect_root()
    cfg = base.load_json(base.join_path(root, "k230_config.json"))

    runtime_cfg = cfg.get("runtime", {})
    cli_args = []
    if sys is not None:
        try:
            cli_args = list(sys.argv[1:])
        except Exception:
            cli_args = []
    compare_samples = parse_compare_samples(cli_args, runtime_cfg)
    runtime_cfg["mode"] = "csv_cached"
    runtime_cfg["max_samples"] = compare_samples
    runtime_cfg["write_predictions_csv"] = True
    cfg["runtime"] = runtime_cfg

    uart_cfg = cfg.get("uart", {})
    uart_cfg["enabled"] = False
    cfg["uart"] = uart_cfg

    paths = cfg["paths"]
    paths["predictions_csv"] = OUTPUT_CSV_NAME
    cfg["paths"] = paths

    scaler_json_path = base.join_path(root, paths["scaler_json"])
    kmodel_path = base.join_path(root, paths["kmodel"])
    pred_csv = base.join_path(root, paths["predictions_csv"])

    # 离线对比固定从第 0 条样本开始，便于和 PC 侧逐行核对。
    base.RUNTIME_CACHE["cursor"] = 0

    t_start = base.now_us()
    X_scaled, y_all, rebuilt = base.ensure_dataset_cache(
        cfg=cfg,
        root=root,
        max_samples=compare_samples,
        scaler_json_path=scaler_json_path,
    )
    if rebuilt:
        print("dataset_cache_rebuilt_samples:", int(X_scaled.shape[0]))
    else:
        print("dataset_cache_hit_samples:", int(X_scaled.shape[0]))

    total_samples = int(X_scaled.shape[0])
    count = total_samples if compare_samples is None else int(compare_samples)
    if count > total_samples:
        count = total_samples
    if count <= 0:
        raise RuntimeError("No cached samples available for comparison.")
    runtime_cfg["infer_batch_size"] = count

    start_idx = 0
    y_batch = base.collect_labels_range(y_all, start_idx, count)
    y_pred, infer_us, model_reloaded = base.run_kmodel_inference_cached(
        kmodel_path=kmodel_path,
        X_scaled=X_scaled,
        start_idx=start_idx,
        count=count,
        uart_sender=None,
    )
    base.write_predictions(pred_csv, y_batch, y_pred)

    t_end = base.now_us()
    total_us = base.diff_us(t_end, t_start)
    mae = base.safe_metric_mae(y_batch, y_pred)
    rmse = base.safe_metric_rmse(y_batch, y_pred)

    print("=== K230 Raw+CNN Compare ===")
    print("root:", root)
    print("mode:", "csv_cached")
    print("kmodel:", kmodel_path)
    print("dataset_total_samples:", total_samples)
    print("infer_batch_size:", count)
    print("infer_start_idx:", start_idx)
    print("model_reloaded:", bool(model_reloaded))
    print("input_shape:", tuple(X_scaled.shape[1:]))
    print("model_infer_time_sec:", infer_us / 1_000_000.0)
    print("model_infer_time_per_sample_ms:", infer_us / 1000.0 / float(count))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("prediction_csv:", pred_csv)
    print("first_10_predictions:", y_pred[:10].tolist())
    if compare_samples is None:
        print("compare_samples:", "all")
    else:
        print("compare_samples:", int(compare_samples))


if __name__ == "__main__":
    main()
