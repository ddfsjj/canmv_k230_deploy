"""统一输出映射和异常码合成。"""

from runtime import protocol


def _finite_or_default(value, default_value):
    """中文注释：把 NaN/Inf 或不可转浮点的值替换成默认值。"""
    try:
        if protocol.is_finite_number(value):
            return float(value)
    except Exception:
        pass
    return float(default_value)


def sanitize_output_value(value, guard_cfg):
    """中文注释：按输出保护配置清洗单个预测值，避免异常大值直接进入返回帧。"""
    if not isinstance(guard_cfg, dict) or not bool(guard_cfg.get("enabled", False)):
        return float(value)

    default_value = float(guard_cfg.get("replace_non_finite_with", 0.0))
    out = _finite_or_default(value, default_value)

    min_value = guard_cfg.get("min", None)
    if min_value is not None and out < float(min_value):
        out = float(min_value)

    max_value = guard_cfg.get("max", None)
    if max_value is not None and out > float(max_value):
        out = float(max_value)

    return out


def sanitize_output_values(values, output_cfg):
    """中文注释：统一保护 12 路输出值，普通返回和异常码返回都走这里。"""
    guard_cfg = output_cfg.get("value_guard", {})
    if not isinstance(guard_cfg, dict) or not bool(guard_cfg.get("enabled", False)):
        return values
    return [sanitize_output_value(value, guard_cfg) for value in values]


def build_values_from_prediction_map(model_pred_map, value_count, output_cfg):
    """中文注释：按输出槽位配置，把命名预测结果映射成固定长度返回值列表。"""
    fill_value = float(output_cfg.get("fill_value", 0.0))
    slots = output_cfg.get("slots", [])
    values = [fill_value] * int(value_count)
    for i in range(len(values)):
        if i >= len(slots):
            break
        output_name = slots[i]
        if output_name is None:
            continue
        output_name = str(output_name)
        if output_name in model_pred_map:
            values[i] = float(model_pred_map[output_name])
    return sanitize_output_values(values, output_cfg)


def build_slot_values(predictions, slot_count, default_value=0.0):
    """中文注释：把命名预测值映射到固定槽位。"""
    values = [float(default_value)] * int(slot_count)
    for pred in predictions:
        if not getattr(pred, "ready", True):
            continue
        slot = int(getattr(pred, "output_slot", 0))
        if 0 <= slot < len(values):
            values[slot] = float(getattr(pred, "value", 0.0))
    return values


def pack_alarm_values(values, error_codes, scale):
    """中文注释：把异常码和干度值合成当前协议的 int32 返回值。"""
    packed = []
    for i, value in enumerate(values):
        code = 0
        if i < len(error_codes):
            code = int(error_codes[i])
        packed.append(protocol.pack_alarm_dryness(code, value, dryness_scale=scale))
    return packed


def build_slot_error_codes(output_cfg, output_source_by_name, channel_raw_error_codes, full_gas_alarm, value_count):
    """中文注释：按输出槽位找到来源物理通道，再为每个返回槽生成异常码。"""
    slots = output_cfg.get("slots", [])
    codes = []
    for i in range(int(value_count)):
        code = protocol.RAW_ANOMALY_OK
        output_name = None
        if i < len(slots) and slots[i] is not None:
            output_name = str(slots[i])
            if output_name in output_source_by_name:
                source_index = int(output_source_by_name[output_name])
                if source_index >= 0 and source_index < len(channel_raw_error_codes):
                    code = int(channel_raw_error_codes[source_index])
        if (
            code == protocol.RAW_ANOMALY_OK
            and getattr(full_gas_alarm, "enabled", False)
            and getattr(full_gas_alarm, "alarm_on", False)
            and output_name in output_source_by_name
        ):
            code = protocol.FULL_GAS_ALARM_CODE
        codes.append(code)
    return codes


def send_uart_prediction_frame(
    uart_sender,
    model_pred_map,
    output_cfg,
    output_source_by_name,
    channel_raw_error_codes,
    raw_anomaly_enabled,
    full_gas_alarm,
):
    """中文注释：统一发送 12 路预测帧，包含普通输出和异常码打包两种路径。"""
    value_count = int(uart_sender.value_count)
    tx_values = build_values_from_prediction_map(model_pred_map, value_count, output_cfg)
    full_gas_enabled = bool(getattr(full_gas_alarm, "enabled", False))
    if raw_anomaly_enabled or full_gas_enabled:
        error_codes = build_slot_error_codes(
            output_cfg=output_cfg,
            output_source_by_name=output_source_by_name,
            channel_raw_error_codes=channel_raw_error_codes,
            full_gas_alarm=full_gas_alarm,
            value_count=value_count,
        )
        packed_values = pack_alarm_values(tx_values, error_codes, uart_sender.scale)
        uart_sender.send_raw_int_values_frame(packed_values)
        return tx_values, error_codes

    uart_sender.send_values_frame(tx_values)
    return tx_values, None
