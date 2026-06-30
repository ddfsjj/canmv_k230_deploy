"""运行配置读取、校验和旧后端适配。"""

import json


DEFAULT_RUNTIME_CONFIG_PATH = "configs/runtime.json"


def _copy_dict(value):
    """中文注释：用 JSON 往返做简单深拷贝，兼容 MicroPython 常用数据类型。"""
    return json.loads(json.dumps(value))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_runtime_config(path):
    cfg = load_json(path)
    validate_runtime_config_dict(cfg)
    return cfg


def is_new_runtime_config(cfg):
    return isinstance(cfg, dict) and "input" in cfg and "models" in cfg and "output" in cfg


def validate_runtime_config_dict(cfg):
    """中文注释：做板端可承受的轻量校验，完整文件存在性由 PC 脚本检查。"""
    if not isinstance(cfg, dict):
        raise ValueError("runtime config must be a JSON object.")
    if is_new_runtime_config(cfg):
        if "version" not in cfg:
            raise ValueError("runtime config missing version.")
        input_cfg = cfg.get("input", {})
        channel_count = int(input_cfg.get("channel_count", 12))
        if channel_count <= 0:
            raise ValueError("input.channel_count must be > 0.")
        models = cfg.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError("runtime config models must be a non-empty list.")
        output_cfg = cfg.get("output", {})
        slot_count = int(output_cfg.get("slot_count", 12))
        if slot_count <= 0:
            raise ValueError("output.slot_count must be > 0.")
        used_slots = {}
        for idx, model in enumerate(models):
            if not isinstance(model, dict):
                raise ValueError("models[{}] must be an object.".format(idx))
            if not bool(model.get("enabled", True)):
                continue
            name = str(model.get("name", "")).strip()
            if not name:
                raise ValueError("models[{}].name is required.".format(idx))
            input_channels = model.get("input_channels", [0])
            if not isinstance(input_channels, list) or not input_channels:
                raise ValueError("models[{}].input_channels must be a non-empty list.".format(idx))
            for ch in input_channels:
                ch_idx = int(ch)
                if ch_idx < 0 or ch_idx >= channel_count:
                    raise ValueError("models[{}].input_channels contains out-of-range channel {}.".format(idx, ch_idx))
            output = model.get("output", {})
            output_slots = output.get("slots", None)
            if output_slots is None:
                output_slots = [output.get("slot", idx)]
            if isinstance(output_slots, dict):
                slot_items = output_slots.items()
            else:
                slot_items = []
                for pos, slot_value in enumerate(output_slots):
                    channel_key = input_channels[pos] if pos < len(input_channels) else pos
                    slot_items.append((channel_key, slot_value))
            for channel_key, slot_value in slot_items:
                slot = int(slot_value)
                if slot < 0 or slot >= slot_count:
                    raise ValueError("models[{}].output slot out of range.".format(idx))
                output_label = "{}:{}".format(name, channel_key)
                if slot in used_slots:
                    raise ValueError("output slot {} is used by both {} and {}.".format(slot, used_slots[slot], output_label))
                used_slots[slot] = output_label
            assets = model.get("assets", {})
            if not assets.get("kmodel") or not assets.get("scaler_json"):
                raise ValueError("models[{}].assets.kmodel and scaler_json are required.".format(idx))
        return

    # 中文注释：旧配置仍允许进入，便于兼容 k230_config_multi.json。
    if "runtime" not in cfg:
        raise ValueError("legacy config missing runtime section.")


def _model_input_names(input_channels):
    names = []
    for ch in input_channels:
        names.append("ch{}".format(int(ch)))
    return names


def to_legacy_multi_config(cfg):
    """中文注释：把新版唯一配置转换成旧多模型后端可直接运行的配置。"""
    if not is_new_runtime_config(cfg):
        return _copy_dict(cfg)

    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    status_cfg = cfg.get("status", {})
    runtime_cfg = cfg.get("runtime", {})
    uart_input_cfg = input_cfg.get("uart", {})
    output_frame_cfg = output_cfg.get("frame", {})

    channel_count = int(input_cfg.get("channel_count", 12))
    slot_count = int(output_cfg.get("slot_count", 12))

    inputs_by_channel = {}
    legacy_models = []
    slots = [None] * slot_count

    for model in cfg.get("models", []):
        if not bool(model.get("enabled", True)):
            continue
        name = str(model.get("name"))
        input_channels = model.get("input_channels", [0])
        for ch in input_channels:
            ch_idx = int(ch)
            inputs_by_channel[ch_idx] = {"name": "ch{}".format(ch_idx), "source_index": ch_idx}

        output_cfg_item = model.get("output", {})
        output_base_name = str(output_cfg_item.get("name", name))
        output_slots = output_cfg_item.get("slots", None)
        if output_slots is None:
            output_slots = [output_cfg_item.get("slot", len(legacy_models))]
        if isinstance(output_slots, dict):
            slot_map = {}
            for key, value in output_slots.items():
                slot_map[int(key)] = int(value)
        else:
            slot_map = {}
            for pos, value in enumerate(output_slots):
                if pos < len(input_channels):
                    slot_map[int(input_channels[pos])] = int(value)
        for ch in input_channels:
            ch_idx = int(ch)
            if len(input_channels) == 1:
                output_name = output_base_name
            else:
                output_name = "{}_ch{}".format(output_base_name, ch_idx)
            if ch_idx in slot_map:
                slots[int(slot_map[ch_idx])] = output_name

        window_cfg = model.get("window", {})
        legacy_model = {
            "name": name,
            "type": model.get("model_type", model.get("type", "cnn_tcn")),
            "input": _model_input_names(input_channels)[0] if len(input_channels) == 1 else None,
            "inputs": _model_input_names(input_channels) if len(input_channels) > 1 else None,
            "output_name": output_base_name if len(input_channels) == 1 else None,
            "paths": {
                "kmodel": model.get("assets", {}).get("kmodel"),
                "scaler_json": model.get("assets", {}).get("scaler_json"),
            },
            "data": {
                "base_window_size": int(window_cfg.get("base_window_size", 500)),
                "base_step": int(window_cfg.get("base_step", 200)),
                "sequence_length": int(window_cfg.get("sequence_length", 5)),
                "sequence_step": int(window_cfg.get("sequence_step", 1)),
            },
            "preprocessing": {
                "feature_mode": window_cfg.get("feature_mode", model.get("feature_mode", "window_demean")),
            },
        }
        cleaned = {}
        for key, value in legacy_model.items():
            if value is not None:
                cleaned[key] = value
        legacy_models.append(cleaned)

    inputs = []
    for ch in sorted(inputs_by_channel):
        inputs.append(inputs_by_channel[ch])
    if not inputs:
        inputs = [{"name": "ch0", "source_index": 0}]

    legacy_runtime = {
        "mode": runtime_cfg.get("mode", input_cfg.get("type", "uart_online")),
        "uart_online": {
            "channel_count": channel_count,
            "input_value_type": uart_input_cfg.get("value_type", "int32"),
            "input_byte_order": uart_input_cfg.get("byte_order", "big"),
            "idle_sleep_ms": int(runtime_cfg.get("idle_sleep_ms", 1)),
            "send_zeros_before_ready": bool(runtime_cfg.get("send_zeros_before_ready", False)),
            "quiet": bool(runtime_cfg.get("quiet", False)),
            "debug_predict_trace": bool(runtime_cfg.get("debug_predict_trace", True)),
            "debug_uart_read_timing": bool(runtime_cfg.get("debug_uart_read_timing", False)),
            "debug_outer_rx": bool(runtime_cfg.get("debug_outer_rx", False)),
            "debug_outer_rx_only_abnormal": bool(runtime_cfg.get("debug_outer_rx_only_abnormal", False)),
            "debug_outer_rx_interval_warn_ms": float(runtime_cfg.get("debug_outer_rx_interval_warn_ms", 25.0)),
            "debug_tx_timing": bool(runtime_cfg.get("debug_tx_timing", False)),
            "debug_tx_only_abnormal": bool(runtime_cfg.get("debug_tx_only_abnormal", True)),
            "debug_tx_interval_min_warn_ms": float(runtime_cfg.get("debug_tx_interval_min_warn_ms", 180.0)),
            "debug_tx_interval_max_warn_ms": float(runtime_cfg.get("debug_tx_interval_max_warn_ms", 240.0)),
            "flush_rx_on_start": bool(runtime_cfg.get("flush_rx_on_start", True)),
            "startup_flush_empty_rounds": int(runtime_cfg.get("startup_flush_empty_rounds", 3)),
            "startup_flush_sleep_ms": int(runtime_cfg.get("startup_flush_sleep_ms", 10)),
        },
        "output": {
            "fill_value": float(output_cfg.get("default_value", 0.0)),
            "slots": slots,
            "value_guard": output_cfg.get("value_guard", {"enabled": False}),
        },
        "full_gas_alarm": status_cfg.get("full_gas_alarm", {"enabled": False}),
        "csv_cached": runtime_cfg.get(
            "csv_cached",
            {"max_samples": 12, "infer_batch_size": 12, "write_predictions_csv": True},
        ),
    }

    uart_cfg = _copy_dict(uart_input_cfg)
    for key, value in output_frame_cfg.items():
        uart_cfg[key] = value
    uart_cfg["enabled"] = bool(uart_cfg.get("enabled", input_cfg.get("type", "uart") == "uart"))
    uart_cfg["value_count"] = int(output_cfg.get("slot_count", uart_cfg.get("value_count", 12)))
    uart_cfg["predict_scale"] = float(output_cfg.get("predict_scale", uart_cfg.get("predict_scale", 100)))
    uart_cfg["value_type"] = output_cfg.get("value_type", uart_cfg.get("value_type", "int32"))
    uart_cfg["byte_order"] = output_cfg.get("byte_order", uart_cfg.get("byte_order", "big"))

    return {
        "name": cfg.get("profile_name", "runtime"),
        "paths": {
            "test_data_dir": input_cfg.get("csv", {}).get("path", cfg.get("paths", {}).get("test_data_dir", "data")),
            "predictions_csv": output_cfg.get("predictions_csv", "predictions_runtime.csv"),
        },
        "inputs": inputs,
        "zero_guard": status_cfg.get("zero_guard", {"enabled": False}),
        "raw_anomaly_alarm": status_cfg.get("raw_anomaly", {"enabled": False}),
        "postprocessing": status_cfg.get("postprocessing", {"enabled": False, "type": "none"}),
        "runtime": legacy_runtime,
        "uart": uart_cfg,
        "models": legacy_models,
    }
