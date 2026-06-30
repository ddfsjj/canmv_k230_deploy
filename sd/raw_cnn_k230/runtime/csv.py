"""CSV 缓存推理后端。

这里用于板端或本地离线调试：统一生成共享样本，再按各模型需要的
sequence_length 裁剪、标准化并推理。UART 在线主路径不依赖这个模块。
"""

import gc

import run_k230_infer as base
from runtime import bindings as runtime_bindings


CSV_RUNTIME_CACHE = {
    "dataset_key": None,
    "X_shared": None,
    "y": None,
    "cursor": 0,
}


def make_shared_dataset_cache_key(root, test_data_dir, model_contexts, max_samples):
    """中文注释：给 CSV 共享样本缓存生成键，关键输入变化时自动重建。"""
    window_size, base_step, feature_mode, max_seq_length = runtime_bindings.get_common_runtime_shape(model_contexts)
    files = base.list_csv_files(test_data_dir)
    parts = [
        base.norm_path(root),
        base.norm_path(test_data_dir),
        str(window_size),
        str(base_step),
        str(feature_mode),
        str(max_seq_length),
        str(max_samples),
    ]
    for path in files:
        size, mtime = base.file_size_mtime(path)
        parts.append("{}:{}:{}".format(base.norm_path(path), size, mtime))
    return "|".join(parts)


def build_shared_dataset(cfg, root, model_contexts, max_samples):
    """中文注释：按最大序列长度构造共享样本，后续各模型再自行裁剪。"""
    paths_cfg = runtime_bindings.require_field(cfg, "paths", "config")
    test_data_dir = runtime_bindings.require_field(paths_cfg, "test_data_dir", "config.paths")
    window_size, base_step, feature_mode, max_seq_length = runtime_bindings.get_common_runtime_shape(model_contexts)
    shared_cfg = {
        "paths": {
            "test_data_dir": test_data_dir,
        },
        "data": {
            "base_window_size": window_size,
            "base_step": base_step,
            "sequence_length": max_seq_length,
            "sequence_step": 1,
        },
        "preprocessing": {
            "feature_mode": feature_mode,
        },
    }
    return base.build_dataset(shared_cfg, root=root, max_samples=max_samples)


def ensure_shared_dataset_cache(cfg, root, model_contexts, max_samples):
    """中文注释：构造或命中 CSV 共享样本缓存。"""
    paths_cfg = runtime_bindings.require_field(cfg, "paths", "config")
    test_data_dir = base.join_path(root, runtime_bindings.require_field(paths_cfg, "test_data_dir", "config.paths"))
    cache_key = make_shared_dataset_cache_key(root, test_data_dir, model_contexts, max_samples)
    if (
        CSV_RUNTIME_CACHE.get("dataset_key", None) == cache_key
        and CSV_RUNTIME_CACHE.get("X_shared", None) is not None
        and CSV_RUNTIME_CACHE.get("y", None) is not None
    ):
        return CSV_RUNTIME_CACHE["X_shared"], CSV_RUNTIME_CACHE["y"], False

    X_shared, y = build_shared_dataset(cfg, root, model_contexts, max_samples)
    CSV_RUNTIME_CACHE["dataset_key"] = cache_key
    CSV_RUNTIME_CACHE["X_shared"] = X_shared
    CSV_RUNTIME_CACHE["y"] = y
    CSV_RUNTIME_CACHE["cursor"] = 0
    return X_shared, y, True


def acquire_infer_range(total_samples, request_count):
    """中文注释：CSV 模式按游标抓取下一批样本，支持循环回绕。"""
    if total_samples <= 0:
        raise RuntimeError("No cached shared samples available.")
    count = int(request_count)
    if count <= 0:
        count = 1
    if count > total_samples:
        count = total_samples

    start_idx = int(CSV_RUNTIME_CACHE.get("cursor", 0)) % int(total_samples)
    next_cursor = start_idx + count
    while next_cursor >= total_samples:
        next_cursor -= total_samples
    CSV_RUNTIME_CACHE["cursor"] = next_cursor
    return start_idx, count


def collect_labels_range(y_all, start_idx, count):
    """中文注释：按游标范围取出当前批次标签。"""
    total = int(len(y_all))
    out = base.empty_float((count,))
    idx = int(start_idx)
    for i in range(count):
        out[i] = y_all[idx]
        idx += 1
        if idx >= total:
            idx = 0
    return out


def collect_shared_sample_range(X_shared, start_idx, count):
    """中文注释：按游标范围取出当前批次共享样本。"""
    total = int(X_shared.shape[0])
    seq_len = int(X_shared.shape[1])
    width = int(X_shared.shape[2])
    out = base.empty_float((count, seq_len, width))
    idx = int(start_idx)
    for i in range(count):
        out[i] = X_shared[idx]
        idx += 1
        if idx >= total:
            idx = 0
    return out


def write_predictions(path, y_true, pred_map, model_names):
    """中文注释：输出多模型合并预测 CSV，方便横向对比。"""
    out_path = base.norm_path(path)
    out_dir = base.dirname(out_path)
    if out_dir not in {"", "."}:
        base.ensure_dir(out_dir)
    with open(out_path, "w") as f:
        header = ["sample_id", "true_label"]
        for model_name in model_names:
            header.append("pred_{}".format(model_name))
        f.write(",".join(header) + "\n")
        for i in range(len(y_true)):
            row = [str(i), str(float(y_true[i]))]
            for model_name in model_names:
                row.append(str(float(pred_map[model_name][i])))
            f.write(",".join(row) + "\n")


def run_csv_cached(cfg, root):
    """中文注释：统一 CSV 缓存推理入口。"""
    runtime_cfg = cfg.get("runtime", {})
    csv_cfg = base.get_runtime_section(runtime_cfg, "csv_cached")
    model_cfgs = cfg.get("models", [])
    model_contexts = [runtime_bindings.ModelRuntimeContext(root, item) for item in model_cfgs]
    runtime_bindings.validate_multi_models(model_contexts)

    paths_cfg = runtime_bindings.require_field(cfg, "paths", "config")
    pred_csv = base.join_path(root, runtime_bindings.require_field(paths_cfg, "predictions_csv", "config.paths"))

    max_samples = csv_cfg.get("max_samples", runtime_cfg.get("max_samples", None))
    if max_samples is not None:
        max_samples = base.require_positive_int(max_samples, "runtime.csv_cached.max_samples")

    infer_batch_size = int(csv_cfg.get("infer_batch_size", 12))
    if infer_batch_size <= 0:
        infer_batch_size = 1

    write_csv = bool(csv_cfg.get("write_predictions_csv", True))

    t_start = base.now_us()
    X_shared, y_all, rebuilt = ensure_shared_dataset_cache(cfg, root, model_contexts, max_samples)
    if rebuilt:
        print("dataset_cache_rebuilt_samples:", int(X_shared.shape[0]))
    else:
        print("dataset_cache_hit_samples:", int(X_shared.shape[0]))

    start_idx, count = acquire_infer_range(int(X_shared.shape[0]), infer_batch_size)
    y_batch = collect_labels_range(y_all, start_idx, count)
    X_batch = collect_shared_sample_range(X_shared, start_idx, count)

    pred_map = {}
    infer_us_map = {}
    for ctx in model_contexts:
        preds = []
        infer_us_total = 0
        scaled_sample = base.empty_float((ctx.sequence_length, ctx.window_size))
        for i in range(count):
            model_sample = runtime_bindings.adapt_shared_sample_for_model(ctx, X_batch[i])
            model_sample = runtime_bindings.scale_sample_for_model(ctx, model_sample, scaled_sample)
            pred, infer_us = runtime_bindings.run_prebuilt_sample(ctx, model_sample)
            preds.append(pred)
            infer_us_total += infer_us
            if (i + 1) % 64 == 0:
                gc.collect()
        pred_map[ctx.name] = base.as_float_array(preds)
        infer_us_map[ctx.name] = infer_us_total

    if write_csv:
        write_predictions(pred_csv, y_batch, pred_map, [ctx.name for ctx in model_contexts])

    t_end = base.now_us()
    total_us = base.diff_us(t_end, t_start)

    print("=== K230 Unified CSV Runtime ===")
    print("root:", root)
    print("config_name:", cfg.get("name", ""))
    print("mode:", "csv_cached")
    print("dataset_total_samples:", int(X_shared.shape[0]))
    print("shared_input_shape:", tuple(X_shared.shape[1:]))
    print("infer_batch_size:", int(count))
    print("infer_start_idx:", int(start_idx))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("write_predictions_csv:", bool(write_csv))
    if write_csv:
        print("prediction_csv:", pred_csv)

    for ctx in model_contexts:
        preds = pred_map[ctx.name]
        infer_us = infer_us_map[ctx.name]
        print("--- model:", ctx.name, "---")
        print("kmodel:", ctx.kmodel_path)
        print("input_shape:", (ctx.sequence_length, ctx.window_size))
        print("model_infer_time_sec:", infer_us / 1_000_000.0)
        print("model_infer_time_per_sample_ms:", infer_us / 1000.0 / float(count))
        print("MAE:", base.safe_metric_mae(y_batch, preds))
        print("RMSE:", base.safe_metric_rmse(y_batch, preds))
        print("first_10_predictions:", preds[:10].tolist())
