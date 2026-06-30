"""在 PC 上仿真 runtime 输出映射和返回帧。

这个脚本不运行 KPU，只验证 runtime.json 展开后的 named prediction、slot、
异常码合成和 52 字节内层返回帧是否符合预期。
"""

import argparse
import json
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K230_DIR = ROOT / "raw_cnn_k230"
if str(K230_DIR) not in sys.path:
    sys.path.insert(0, str(K230_DIR))

from runtime import output as runtime_output  # noqa: E402
from runtime import protocol as runtime_protocol  # noqa: E402
from runtime.config import is_new_runtime_config, load_runtime_config, to_legacy_multi_config  # noqa: E402


class FullGasAlarmStub:
    def __init__(self, enabled=False, alarm_on=False):
        self.enabled = bool(enabled)
        self.alarm_on = bool(alarm_on)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate K230 runtime output slots and UART frame.")
    parser.add_argument(
        "--config",
        default="raw_cnn_k230/configs/runtime.json",
        help="Runtime config path, relative to repo root by default.",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated slot values. If fewer than 12, remaining slots use config defaults.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        help="Named prediction override, for example --pred model_1_cnn_tcn_ch0=0.23.",
    )
    parser.add_argument(
        "--raw-errors",
        default="",
        help="Physical channel error codes, for example 0=1,1=4.",
    )
    parser.add_argument(
        "--full-gas",
        action="store_true",
        help="Simulate full gas alarm enabled and active.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_float_list(text):
    if text is None or str(text).strip() == "":
        return None
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def parse_prediction_overrides(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--pred must use name=value format: {}".format(item))
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--pred name must not be empty.")
        out[name] = float(value)
    return out


def parse_raw_errors(text, channel_count):
    codes = [runtime_protocol.RAW_ANOMALY_OK] * int(channel_count)
    if not text:
        return codes
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--raw-errors must use channel=code format: {}".format(item))
        channel, code = item.split("=", 1)
        idx = int(channel.strip())
        if idx < 0 or idx >= len(codes):
            raise ValueError("raw error channel out of range: {}".format(idx))
        codes[idx] = int(code.strip())
    return codes


def build_output_source_map(cfg):
    """中文注释：从新 runtime 配置还原 named prediction 对应的物理通道。"""
    result = {}
    if not is_new_runtime_config(cfg):
        return result
    for model in cfg.get("models", []):
        if not bool(model.get("enabled", True)):
            continue
        model_name = str(model.get("name", "model"))
        input_channels = [int(v) for v in model.get("input_channels", [0])]
        output_cfg = model.get("output", {})
        output_base_name = str(output_cfg.get("name", model_name))
        for ch in input_channels:
            if len(input_channels) == 1:
                output_name = output_base_name
            else:
                output_name = "{}_ch{}".format(output_base_name, ch)
            result[output_name] = ch
    return result


def default_prediction_map(slots):
    pred_map = {}
    next_value = 0.11
    for name in slots:
        if name is None:
            continue
        text = str(name)
        if text not in pred_map:
            pred_map[text] = round(next_value, 6)
            next_value += 0.01
    return pred_map


def values_to_prediction_map(slot_values, slots):
    pred_map = {}
    if slot_values is None:
        return pred_map
    for idx, value in enumerate(slot_values):
        if idx >= len(slots):
            break
        if slots[idx] is None:
            continue
        pred_map[str(slots[idx])] = float(value)
    return pred_map


def encode_inner_frame(values, uart_cfg, apply_scale):
    header = bytearray(int(v) & 0xFF for v in uart_cfg.get("header", [0x55, 0xAA]))
    tail = bytearray(int(v) & 0xFF for v in uart_cfg.get("tail", [0xFC, 0xCF]))
    value_count = int(uart_cfg.get("value_count", 12))
    byte_order = str(uart_cfg.get("byte_order", "big")).lower()
    value_type = str(uart_cfg.get("value_type", "int32")).lower()
    scale = float(uart_cfg.get("predict_scale", 100))
    int_fmt = ">i" if byte_order == "big" else "<i"
    float_fmt = ">f" if byte_order == "big" else "<f"

    payload = bytearray()
    for idx in range(value_count):
        value = float(values[idx]) if idx < len(values) else 0.0
        if value_type == "float32":
            payload.extend(struct.pack(float_fmt, value))
        else:
            if apply_scale:
                packed_value = runtime_protocol.clamp_int32(int(round(value * scale)))
            else:
                packed_value = runtime_protocol.clamp_int32(int(round(value)))
            payload.extend(struct.pack(int_fmt, packed_value))
    return bytes(header + payload + tail)


def build_rows(slots, tx_values, error_codes, packed_values, output_source_by_name, value_count):
    rows = []
    for idx in range(int(value_count)):
        name = slots[idx] if idx < len(slots) else None
        source = None
        if name is not None:
            source = output_source_by_name.get(str(name))
        rows.append(
            {
                "slot": idx,
                "name": name,
                "source_channel": source,
                "value": float(tx_values[idx]) if idx < len(tx_values) else 0.0,
                "error_code": int(error_codes[idx]) if error_codes is not None and idx < len(error_codes) else None,
                "packed_int32": int(packed_values[idx]) if packed_values is not None and idx < len(packed_values) else None,
            }
        )
    return rows


def main():
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    cfg = load_runtime_config(str(config_path))
    legacy_cfg = to_legacy_multi_config(cfg)
    runtime_cfg = legacy_cfg.get("runtime", {})
    output_cfg = runtime_cfg.get("output", {})
    uart_cfg = legacy_cfg.get("uart", {})
    value_count = int(uart_cfg.get("value_count", 12))
    scale = float(uart_cfg.get("predict_scale", 100))
    slots = output_cfg.get("slots", [])

    pred_map = default_prediction_map(slots)
    pred_map.update(values_to_prediction_map(parse_float_list(args.values), slots))
    pred_map.update(parse_prediction_overrides(args.pred))

    output_source_by_name = build_output_source_map(cfg)
    channel_count = int(legacy_cfg.get("runtime", {}).get("uart_online", {}).get("channel_count", value_count))
    channel_raw_error_codes = parse_raw_errors(args.raw_errors, channel_count)
    full_gas_alarm = FullGasAlarmStub(enabled=args.full_gas, alarm_on=args.full_gas)
    raw_anomaly_enabled = bool(legacy_cfg.get("raw_anomaly_alarm", {}).get("enabled", False))

    tx_values = runtime_output.build_values_from_prediction_map(
        model_pred_map=pred_map,
        value_count=value_count,
        output_cfg=output_cfg,
    )
    error_codes = None
    packed_values = None
    if raw_anomaly_enabled or bool(args.full_gas):
        error_codes = runtime_output.build_slot_error_codes(
            output_cfg=output_cfg,
            output_source_by_name=output_source_by_name,
            channel_raw_error_codes=channel_raw_error_codes,
            full_gas_alarm=full_gas_alarm,
            value_count=value_count,
        )
        packed_values = runtime_output.pack_alarm_values(
            values=tx_values,
            error_codes=error_codes,
            scale=scale,
        )
        frame = encode_inner_frame(packed_values, uart_cfg, apply_scale=False)
    else:
        frame = encode_inner_frame(tx_values, uart_cfg, apply_scale=True)

    rows = build_rows(slots, tx_values, error_codes, packed_values, output_source_by_name, value_count)
    result = {
        "config": str(config_path),
        "profile": legacy_cfg.get("name", ""),
        "value_count": value_count,
        "predict_scale": scale,
        "slots": slots,
        "prediction_map": pred_map,
        "rows": rows,
        "frame_len": len(frame),
        "frame_hex": frame.hex(" "),
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("=== K230 Runtime Output Simulation ===")
    print("config:", result["config"])
    print("profile:", result["profile"])
    print("value_count:", value_count)
    print("predict_scale:", scale)
    print("raw_anomaly_enabled:", raw_anomaly_enabled)
    print("frame_len:", len(frame))
    print("")
    print("slot | source | error | value | packed_int32 | name")
    print("-----+--------+-------+-------+--------------+------------------------------")
    for row in rows:
        print(
            "{slot:>4} | {source!s:>6} | {error!s:>5} | {value:>5.3f} | {packed!s:>12} | {name}".format(
                slot=row["slot"],
                source="" if row["source_channel"] is None else row["source_channel"],
                error="" if row["error_code"] is None else row["error_code"],
                value=row["value"],
                packed="" if row["packed_int32"] is None else row["packed_int32"],
                name="" if row["name"] is None else row["name"],
            )
        )
    print("")
    print("inner_frame_hex:")
    print(result["frame_hex"])


if __name__ == "__main__":
    main()
