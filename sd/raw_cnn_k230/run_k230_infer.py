"""
K230 侧运行主脚本。

这个文件同时承载三类用途：
1. `csv_cached`：离线调试模式。
   从 `test_data` 读取 CSV，切窗后做推理；首次构建缓存，后续循环复用。
2. `uart_online`：在线推理模式。
   从串口实时接收 12 路输入数据，进入环形缓冲，满窗后逐路推理，再回发 12 路结果。
3. `uart_echo`：串口环路测试模式。
   收到什么字节就原样发回什么字节，用于先验证 UART 通信链路。

因为一个脚本要覆盖“离线调试”和“在线联调”两个阶段，
所以配置与流程都集中在这里，方便后续切换模式时只改 JSON，不改入口文件。
"""

import gc
import json
import time
try:
    import sys  # type: ignore
except ImportError:
    sys = None  # type: ignore

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore

try:
    import ustruct as struct  # type: ignore
except ImportError:
    import struct  # type: ignore

try:
    from machine import UART, FPIOA  # type: ignore
except ImportError:
    UART = None  # type: ignore
    FPIOA = None  # type: ignore

try:
    import ulab.numpy as np  # type: ignore
except ImportError:
    import numpy as np  # type: ignore

NP_FLOAT = getattr(np, "float32", None)
if NP_FLOAT is None:
    NP_FLOAT = getattr(np, "float", None)
if NP_FLOAT is None:
    NP_FLOAT = float

# 大将军平时直接运行这个文件时，默认读取这里指定的板端配置。
# 以后如果要切配置，直接改这一行即可。
DEFAULT_RUNTIME_CONFIG_PATH = "configs/k230_config_cnn_tcn.json"
OVERRIDE_CONFIG_PATH = None

# 下面这一大段公共工具函数虽然看起来多，
# 但核心目标只有两个：
# 1. 让同一份脚本同时覆盖离线 CSV 和串口在线多模式。
# 2. 尽量兼容 CanMV / MicroPython / 普通 Python 的运行差异。

# 进程内缓存：
# 1. 缓存离线模式构建好的样本与标准化结果。
# 2. 缓存已经加载好的 kmodel。
# 3. 保存离线模式下的推理游标，避免每轮都从头开始取样本。
RUNTIME_CACHE = {
    "dataset_key": None,
    "X_scaled": None,
    "y": None,
    "cursor": 0,
    "kmodel_key": None,
    "kpu": None,
    "nn": None,
}


def as_float_array(values):
    # 尽量把输入转换成浮点数组，统一后续数值计算入口。
    try:
        return np.asarray(values, dtype=NP_FLOAT)
    except TypeError:
        return np.asarray(values)


def astype_float_array(arr):
    # 某些运行环境下 astype 的 dtype 兼容性不同，这里统一做一次兜底。
    if not hasattr(arr, "astype"):
        return arr
    try:
        return arr.astype(NP_FLOAT)
    except TypeError:
        return arr


def empty_float(shape):
    # 申请浮点数组；在 ulab 与 numpy 间保持相同调用方式。
    try:
        return np.empty(shape, dtype=NP_FLOAT)
    except TypeError:
        return np.empty(shape)


def is_finite_number(value):
    # 只允许正常有限数参与协议打包和推理结果输出。
    v = float(value)
    if v != v:
        return False
    if v == float("inf") or v == float("-inf"):
        return False
    return True


def clamp_int32(value):
    # 当协议选择 int32 发送时，先把数值限制在 int32 可表达范围内。
    if value > 2147483647:
        return 2147483647
    if value < -2147483648:
        return -2147483648
    return int(value)


RAW_ANOMALY_OK = 0x00
RAW_ANOMALY_ALL_ZERO = 0x01
RAW_ANOMALY_LOW = 0x02
RAW_ANOMALY_HIGH = 0x03
RAW_ANOMALY_SPIKE = 0x04
RAW_ANOMALY_STUCK = 0x05
FULL_GAS_ALARM_CODE = 0x10


def get_raw_anomaly_config(cfg):
    # 中文注释：原始数据异常只判断物理输入通道，不和模型数量绑定。
    raw_cfg = cfg.get("raw_anomaly_alarm", cfg.get("anomaly_alarm", {}))
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    return raw_cfg


def _cfg_float(cfg, name, default_value):
    value = cfg.get(name, default_value)
    if value is None:
        return None
    return float(value)


def detect_raw_window_anomaly(raw_window, alarm_cfg):
    # 中文注释：只做明确异常：全 0、硬越界、固定死值、相邻尖峰；不做“小波动”判断。
    if not bool(alarm_cfg.get("enabled", False)):
        return RAW_ANOMALY_OK

    count = int(len(raw_window))
    if count <= 0:
        return RAW_ANOMALY_OK

    first = float(raw_window[0])
    min_v = first
    max_v = first
    all_zero = first == 0.0
    all_same = True
    last = first
    max_diff = 0.0

    for i in range(1, count):
        value = float(raw_window[i])
        if value != 0.0:
            all_zero = False
        if value != first:
            all_same = False
        if value < min_v:
            min_v = value
        if value > max_v:
            max_v = value
        diff = value - last
        if diff < 0:
            diff = -diff
        if diff > max_diff:
            max_diff = diff
        last = value

    if all_zero and bool(alarm_cfg.get("all_zero_enabled", True)):
        return RAW_ANOMALY_ALL_ZERO

    raw_min = _cfg_float(alarm_cfg, "raw_min", None)
    if raw_min is not None and bool(alarm_cfg.get("raw_range_enabled", True)) and min_v < raw_min:
        return RAW_ANOMALY_LOW

    raw_max = _cfg_float(alarm_cfg, "raw_max", None)
    if raw_max is not None and bool(alarm_cfg.get("raw_range_enabled", True)) and max_v > raw_max:
        return RAW_ANOMALY_HIGH

    if all_same and first != 0.0 and bool(alarm_cfg.get("stuck_enabled", True)):
        return RAW_ANOMALY_STUCK

    spike_max_diff = _cfg_float(alarm_cfg, "spike_max_diff", None)
    if spike_max_diff is not None and bool(alarm_cfg.get("spike_enabled", True)) and max_diff > spike_max_diff:
        return RAW_ANOMALY_SPIKE

    return RAW_ANOMALY_OK


def pack_alarm_dryness(error_code, dryness_value, dryness_scale=100.0):
    # 中文注释：4 字节返回格式为 [异常码 1 字节][保留 1 字节][干度 uint16]。
    code = int(error_code) & 0xFF
    dry = 0
    if is_finite_number(dryness_value):
        dry = int(round(float(dryness_value) * float(dryness_scale)))
    if dry < 0:
        dry = 0
    if dry > 65535:
        dry = 65535
    return clamp_int32((code << 24) | int(dry))


class RawChannelAnomalyState:
    # 中文注释：每个物理输入通道独立防抖，异常码跟输入通道走，不跟模型输出走。
    def __init__(self, cfg, channel_count):
        if not isinstance(cfg, dict):
            cfg = {}
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.channel_count = int(channel_count)
        if self.channel_count <= 0:
            self.channel_count = 1
        self.hit_count = clamp_count(cfg.get("hit_count", 1), 1, 1, 64)
        self.clear_count = clamp_count(cfg.get("clear_count", 1), 1, 1, 64)
        self._alarm_on = [False] * self.channel_count
        self._alarm_code = [RAW_ANOMALY_OK] * self.channel_count
        self._hit_counts = [0] * self.channel_count
        self._clear_counts = [0] * self.channel_count

    def update(self, channel, raw_window):
        idx = int(channel)
        if idx < 0 or idx >= self.channel_count:
            return RAW_ANOMALY_OK
        if not self.enabled:
            return RAW_ANOMALY_OK

        code = detect_raw_window_anomaly(raw_window, self.cfg)
        if code != RAW_ANOMALY_OK:
            if self._alarm_code[idx] == code:
                self._hit_counts[idx] += 1
            else:
                self._alarm_code[idx] = code
                self._hit_counts[idx] = 1
            self._clear_counts[idx] = 0
            if self._hit_counts[idx] >= self.hit_count:
                self._alarm_on[idx] = True
            if self._alarm_on[idx]:
                return self._alarm_code[idx]
            return RAW_ANOMALY_OK

        self._hit_counts[idx] = 0
        self._clear_counts[idx] += 1
        if self._clear_counts[idx] >= self.clear_count:
            self._alarm_on[idx] = False
            self._alarm_code[idx] = RAW_ANOMALY_OK
        if self._alarm_on[idx]:
            return self._alarm_code[idx]
        return RAW_ANOMALY_OK

    def codes(self):
        out = []
        for i in range(self.channel_count):
            if self._alarm_on[i]:
                out.append(int(self._alarm_code[i]))
            else:
                out.append(RAW_ANOMALY_OK)
        return out


class UartDrynessSender:
    """
    统一封装 UART 的发送侧能力。

    这个类只关心“如何初始化 UART”和“如何把一组数值打包发送”：
    1. 支持帧头/帧尾自定义。
    2. 支持 int32 / float32 两种 4 字节数值类型。
    3. 支持大小端切换。
    4. 支持把逐个预测值先缓存，攒够 12 路后再发一帧。
    """

    def __init__(self, uart_cfg):
        self.enabled = False
        self.uart = None
        self.quiet = bool(uart_cfg.get("quiet", False))
        self.send_count = 0
        self.error_count = 0
        self.pending_values = []
        self.scale = float(uart_cfg.get("predict_scale", 1000))
        self.value_count = int(uart_cfg.get("value_count", 12))
        self.byte_order = str(uart_cfg.get("byte_order", "little")).lower()
        self.value_type = str(uart_cfg.get("value_type", "int32")).lower()
        self.header = self._parse_frame_bytes(uart_cfg.get("header", [0x55, 0xAA]), [0x55, 0xAA])
        self.tail = self._parse_frame_bytes(uart_cfg.get("tail", [0xFC, 0xCF]), [0xFC, 0xCF])
        self.outer_frame_enabled = bool(uart_cfg.get("outer_frame_enabled", False))
        self.outer_frame_count = int(uart_cfg.get("outer_frame_count", 10))
        self.outer_header = self._parse_frame_bytes(uart_cfg.get("outer_header", [0xF7, 0x7F]), [0xF7, 0x7F])
        self.outer_tail = self._parse_frame_bytes(uart_cfg.get("outer_tail", [0xFA, 0xAF]), [0xFA, 0xAF])
        if self.byte_order not in {"little", "big"}:
            self.byte_order = "little"
        if self.value_type not in {"int32", "float32"}:
            self.value_type = "int32"
        if self.outer_frame_count <= 0:
            self.outer_frame_count = 10
        self.inner_frame_len = len(self.header) + self.value_count * 4 + len(self.tail)
        self.outer_payload_len = self.outer_frame_count * self.inner_frame_len
        self.outer_frame_len = len(self.outer_header) + self.outer_payload_len + len(self.outer_tail)

        # 配置里关闭 UART 时，允许脚本继续运行，只是不做串口发送。
        if not bool(uart_cfg.get("enabled", False)):
            return
        if UART is None or FPIOA is None:
            if not self.quiet:
                print("WARN: machine UART/FPIOA not available, UART send disabled.")
            return
        try:
            uart_id = int(uart_cfg.get("uart_id", 1))
            tx_pin = int(uart_cfg.get("tx_pin", 3))
            rx_pin = int(uart_cfg.get("rx_pin", 4))
            baudrate = int(uart_cfg.get("baudrate", 921600))
            bits = int(uart_cfg.get("bits", 8))
            parity = uart_cfg.get("parity", "none")
            stop = int(uart_cfg.get("stop", 1))

            if bits == 7:
                bits_const = UART.SEVENBITS
            else:
                bits_const = UART.EIGHTBITS

            parity_key = str(parity).lower()
            if parity_key == "even":
                parity_const = UART.PARITY_EVEN
            elif parity_key == "odd":
                parity_const = UART.PARITY_ODD
            else:
                parity_const = UART.PARITY_NONE

            stop_const = UART.STOPBITS_TWO if stop == 2 else UART.STOPBITS_ONE

            # FPIOA 负责把实际物理引脚复用到 UART 功能。
            fpioa = FPIOA()
            tx_func = getattr(fpioa, "UART{}_TXD".format(uart_id))
            rx_func = getattr(fpioa, "UART{}_RXD".format(uart_id))
            try:
                fpioa.set_function(tx_pin, tx_func, ie=1, oe=1)
            except TypeError:
                fpioa.set_function(tx_pin, tx_func)
            try:
                fpioa.set_function(rx_pin, rx_func, ie=1, oe=1)
            except TypeError:
                fpioa.set_function(rx_pin, rx_func)

            uart_const = getattr(UART, "UART{}".format(uart_id), uart_id)
            self.uart = UART(
                uart_const,
                baudrate=baudrate,
                bits=bits_const,
                parity=parity_const,
                stop=stop_const,
            )
            self.enabled = True
            if not self.quiet:
                print(
                    "UART sender enabled: UART{}, {} bps, tx_pin={}, rx_pin={}, value_count={}, byte_order={}, value_type={}, outer_frame_enabled={}, outer_frame_count={}".format(
                        uart_id,
                        baudrate,
                        tx_pin,
                        rx_pin,
                        self.value_count,
                        self.byte_order,
                        self.value_type,
                        self.outer_frame_enabled,
                        self.outer_frame_count,
                    )
                )
        except Exception as exc:
            self.enabled = False
            self.uart = None
            if not self.quiet:
                print("WARN: UART sender init failed, UART send disabled:", exc)

    def _parse_frame_bytes(self, raw, default_bytes):
        # 兼容多种配置写法：
        # 1. [85, 170]
        # 2. "55 AA"
        # 3. 单个整数
        if isinstance(raw, (list, tuple)):
            data = bytearray()
            for b in raw:
                data.append(int(b) & 0xFF)
            if len(data) > 0:
                return data
        if isinstance(raw, str):
            text = raw.replace(",", " ").replace("0x", "").replace("0X", "").strip()
            if text:
                parts = [p for p in text.split() if p]
                data = bytearray()
                ok = True
                for p in parts:
                    try:
                        data.append(int(p, 16) & 0xFF)
                    except ValueError:
                        ok = False
                        break
                if ok and len(data) > 0:
                    return data
        try:
            return bytearray([int(raw) & 0xFF])
        except Exception:
            return bytearray(default_bytes)

    def _encode_frame(self, values, apply_scale=True):
        # 把一组数值编码为一整帧串口数据：
        # header + payload(12 * 4 字节) + tail。
        payload = bytearray()
        count = self.value_count
        int_fmt = ">i" if self.byte_order == "big" else "<i"
        float_fmt = ">f" if self.byte_order == "big" else "<f"
        for i in range(count):
            fval = 0.0
            if i < len(values):
                raw = values[i]
                if is_finite_number(raw):
                    fval = float(raw)
            if self.value_type == "float32":
                payload.extend(struct.pack(float_fmt, float(fval)))
            else:
                if apply_scale:
                    packed_value = clamp_int32(int(round(float(fval) * self.scale)))
                else:
                    packed_value = clamp_int32(int(round(float(fval))))
                payload.extend(struct.pack(int_fmt, int(packed_value)))
        frame = bytearray(self.header)
        frame.extend(payload)
        frame.extend(self.tail)
        return frame

    def send_scaled_prediction(self, pred_value):
        # 适用于“逐个样本得到预测值”的离线模式。
        # 先把单个预测值压入缓存，攒够 value_count 后再统一发一帧。
        if not self.enabled or self.uart is None:
            return
        v = 0.0
        if is_finite_number(pred_value):
            v = float(pred_value)
        self.pending_values.append(v)
        if len(self.pending_values) < self.value_count:
            return
        values = self.pending_values[: self.value_count]
        del self.pending_values[: self.value_count]
        self._send_values(values)

    def _send_values(self, values):
        # 底层发送函数：假定输入已经是一整组待发送数值。
        frame = self._encode_frame(values, apply_scale=True)
        try:
            self.uart.write(frame)
            self.send_count += 1
        except Exception as exc:
            self.error_count += 1
            if not self.quiet:
                print("WARN: UART send failed:", exc)

    def send_raw_int_values_frame(self, values):
        # 调试 ACK 模式下直接发送原始整数，不再乘 predict_scale。
        if not self.enabled or self.uart is None:
            return
        frame = self._encode_frame(values, apply_scale=False)
        try:
            self.uart.write(frame)
            self.send_count += 1
        except Exception as exc:
            self.error_count += 1
            if not self.quiet:
                print("WARN: UART raw-int send failed:", exc)

    def send_values_frame(self, values):
        # 适用于“已经拿到 12 路完整结果”的场景，例如在线模式。
        if not self.enabled or self.uart is None:
            return
        self._send_values(values)

    def flush_pending(self):
        # 把缓存里剩余但不足 12 路的数据补 0 发出去，避免尾包丢失。
        if not self.enabled or self.uart is None:
            return
        if len(self.pending_values) == 0:
            return
        values = []
        for i in range(self.value_count):
            if i < len(self.pending_values):
                values.append(self.pending_values[i])
            else:
                values.append(0.0)
        self.pending_values = []
        self._send_values(values)


def now_us():
    # 返回微秒时间戳，用于统计推理耗时与总流程耗时。
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.perf_counter() * 1_000_000)


def diff_us(t_end, t_start):
    # 兼容不同运行时下的时间差计算接口。
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(t_end, t_start)
    return t_end - t_start


def sleep_ms(ms):
    # 统一的毫秒级 sleep，避免到处判断 time.sleep_ms 是否存在。
    v = int(ms)
    if v <= 0:
        return
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(v)
    else:
        time.sleep(float(v) / 1000.0)


def drain_uart_rx(uart, empty_rounds=3, sleep_between_ms=10):
    # 在线模式启动前，先把 UART 接收缓冲里残留的旧数据清空。
    # 只有连续多次读不到数据，才认为当前链路已经“干净”。
    total_bytes = 0
    empty_hits = 0
    rounds_need = int(empty_rounds)
    if rounds_need <= 0:
        rounds_need = 1
    sleep_v = int(sleep_between_ms)
    if sleep_v < 0:
        sleep_v = 0
    while empty_hits < rounds_need:
        data = uart.read()
        if data:
            total_bytes += len(data)
            empty_hits = 0
        else:
            empty_hits += 1
            # 中文注释：下位机持续发数时，这里睡眠会让新数据趁机进缓冲，
            # 导致启动清空阶段很难连续读到空缓冲；保留快速空读即可。
            # if sleep_v > 0:
            #     sleep_ms(sleep_v)
    return total_bytes


class UartValueFrameParser:
    """
    在线串口输入解析器。

    作用：
    1. 处理串口可能出现的半包、粘包、错位问题。
    2. 在原始字节流里寻找固定帧头和固定帧尾。
    3. 将一整帧 payload 解析为 12 路数值列表。
    """

    def __init__(self, header, tail, value_count, value_type, byte_order):
        self.header = bytes(header)
        self.tail = bytes(tail)
        self.value_count = int(value_count)
        self.value_type = str(value_type).lower()
        self.byte_order = str(byte_order).lower()
        if self.value_type not in {"int32", "float32"}:
            self.value_type = "float32"
        if self.byte_order not in {"little", "big"}:
            self.byte_order = "big"
        self._buf = bytearray()
        self._payload_len = self.value_count * 4
        self._frame_len = len(self.header) + self._payload_len + len(self.tail)

    def _decode_payload(self, payload):
        # 将 payload 的每 4 字节解释为一个数值。
        values = []
        int_fmt = ">i" if self.byte_order == "big" else "<i"
        float_fmt = ">f" if self.byte_order == "big" else "<f"
        for i in range(self.value_count):
            start = i * 4
            chunk = payload[start : start + 4]
            if self.value_type == "float32":
                values.append(float(struct.unpack(float_fmt, chunk)[0]))
            else:
                values.append(float(struct.unpack(int_fmt, chunk)[0]))
        return values

    def feed(self, data):
        # 串口数据是流，不保证一次 read() 恰好得到一帧。
        # 因此这里把新字节先塞进内部缓冲，再不断尝试抽取完整帧。
        if not data:
            return []
        self._buf.extend(data)
        out = []
        header_len = len(self.header)
        tail_len = len(self.tail)
        while len(self._buf) >= self._frame_len:
            idx = self._buf.find(self.header)
            if idx < 0:
                # 无帧头时，仅保留可能构成下次帧头的尾巴，避免缓冲无限增长。
                keep = max(0, header_len - 1)
                if keep > 0 and len(self._buf) > keep:
                    self._buf = bytearray(self._buf[-keep:])
                elif keep == 0:
                    self._buf = bytearray()
                break
            if idx > 0:
                self._buf = bytearray(self._buf[idx:])
            if len(self._buf) < self._frame_len:
                break
            tail_start = header_len + self._payload_len
            if self._buf[tail_start : tail_start + tail_len] != self.tail:
                # 找到了疑似帧头，但对应位置的帧尾不匹配；
                # 说明当前同步点不可信，丢掉 1 字节继续向后搜索。
                self._buf = bytearray(self._buf[1:])
                continue
            payload = self._buf[header_len:tail_start]
            try:
                values = self._decode_payload(payload)
                out.append(values)
            except Exception:
                pass
            self._buf = bytearray(self._buf[self._frame_len :])
        return out


class UartRawFrameParser:
    """
    原始帧提取器。

    与 `UartValueFrameParser` 不同，这里不关心 payload 里的数值含义，
    只负责从串口字节流中切出一整帧原始字节。
    适合做“收到第 N 帧后，原样回发该帧”这类测试。
    """

    def __init__(self, header, tail, value_count):
        self.header = bytes(header)
        self.tail = bytes(tail)
        self.value_count = int(value_count)
        self._payload_len = self.value_count * 4
        self._frame_len = len(self.header) + self._payload_len + len(self.tail)
        self._buf = bytearray()

    def feed(self, data):
        if not data:
            return []
        self._buf.extend(data)
        out = []
        header_len = len(self.header)
        tail_len = len(self.tail)

        while len(self._buf) >= self._frame_len:
            idx = self._buf.find(self.header)
            if idx < 0:
                keep = max(0, header_len - 1)
                if keep > 0 and len(self._buf) > keep:
                    self._buf = bytearray(self._buf[-keep:])
                elif keep == 0:
                    self._buf = bytearray()
                break
            if idx > 0:
                self._buf = bytearray(self._buf[idx:])
            if len(self._buf) < self._frame_len:
                break

            tail_start = header_len + self._payload_len
            if self._buf[tail_start : tail_start + tail_len] != self.tail:
                self._buf = bytearray(self._buf[1:])
                continue

            frame = bytes(self._buf[: self._frame_len])
            out.append(frame)
            self._buf = bytearray(self._buf[self._frame_len :])
        return out


class UartBundledRawFrameParser:
    """
    大帧解析器。

    用于识别：
    1. 外层大帧头 `F7 7F`
    2. 中间固定包含 N 个原始小帧
    3. 外层大帧尾 `FA AF`

    只有当外层包完整，且内部每个小帧都满足原协议时，才认为收到了一帧有效大帧。
    对 `uart_frame_return` 模式来说，返回的是完整大帧原始字节，便于原样回发。
    """

    def __init__(self, outer_header, outer_tail, inner_header, inner_tail, value_count, outer_frame_count):
        self.outer_header = bytes(outer_header)
        self.outer_tail = bytes(outer_tail)
        self.inner_header = bytes(inner_header)
        self.inner_tail = bytes(inner_tail)
        self.value_count = int(value_count)
        self.outer_frame_count = int(outer_frame_count)
        self._inner_payload_len = self.value_count * 4
        self._inner_frame_len = len(self.inner_header) + self._inner_payload_len + len(self.inner_tail)
        self._outer_payload_len = self.outer_frame_count * self._inner_frame_len
        self._outer_frame_len = len(self.outer_header) + self._outer_payload_len + len(self.outer_tail)
        self._buf = bytearray()

    def _validate_inner_frames(self, payload):
        if len(payload) != self._outer_payload_len:
            return False
        header_len = len(self.inner_header)
        tail_len = len(self.inner_tail)
        for i in range(self.outer_frame_count):
            start = i * self._inner_frame_len
            frame = payload[start : start + self._inner_frame_len]
            if frame[:header_len] != self.inner_header:
                return False
            if frame[self._inner_frame_len - tail_len : self._inner_frame_len] != self.inner_tail:
                return False
        return True

    def feed(self, data):
        if not data:
            return []
        self._buf.extend(data)
        out = []
        header_len = len(self.outer_header)
        tail_len = len(self.outer_tail)

        while len(self._buf) >= self._outer_frame_len:
            idx = self._buf.find(self.outer_header)
            if idx < 0:
                keep = max(0, header_len - 1)
                if keep > 0 and len(self._buf) > keep:
                    self._buf = bytearray(self._buf[-keep:])
                elif keep == 0:
                    self._buf = bytearray()
                break
            if idx > 0:
                self._buf = bytearray(self._buf[idx:])
            if len(self._buf) < self._outer_frame_len:
                break

            tail_start = header_len + self._outer_payload_len
            if self._buf[tail_start : tail_start + tail_len] != self.outer_tail:
                self._buf = bytearray(self._buf[1:])
                continue

            payload = self._buf[header_len:tail_start]
            if not self._validate_inner_frames(payload):
                self._buf = bytearray(self._buf[1:])
                continue

            out.append(bytes(self._buf[: self._outer_frame_len]))
            self._buf = bytearray(self._buf[self._outer_frame_len :])
        return out


class UartBundledValueFrameParser:
    """
    大帧拆小帧并解码数值的解析器。

    外层先按大帧头尾取包，再把大帧 payload 按固定长度切成多个原始小帧，
    最后继续复用小帧的数值解码逻辑。这样单片机即使改成 10ms 发 10 帧，
    K230 内部仍然能按“每个小帧是一组 12 路采样值”来处理。
    """

    def __init__(
        self,
        outer_header,
        outer_tail,
        inner_header,
        inner_tail,
        value_count,
        value_type,
        byte_order,
        outer_frame_count,
    ):
        self.outer_header = bytes(outer_header)
        self.outer_tail = bytes(outer_tail)
        self.inner_header = bytes(inner_header)
        self.inner_tail = bytes(inner_tail)
        self.outer_frame_count = int(outer_frame_count)
        self.value_count = int(value_count)
        self.value_type = str(value_type).lower()
        self.byte_order = str(byte_order).lower()
        if self.value_type not in {"int32", "float32"}:
            self.value_type = "float32"
        if self.byte_order not in {"little", "big"}:
            self.byte_order = "big"
        self._inner_payload_len = self.value_count * 4
        self._inner_frame_len = len(self.inner_header) + self._inner_payload_len + len(self.inner_tail)
        self._outer_payload_len = self.outer_frame_count * self._inner_frame_len
        self._outer_frame_len = len(self.outer_header) + self._outer_payload_len + len(self.outer_tail)
        self._buf = bytearray()

    def _decode_payload(self, payload):
        values = []
        int_fmt = ">i" if self.byte_order == "big" else "<i"
        float_fmt = ">f" if self.byte_order == "big" else "<f"
        for i in range(self.value_count):
            start = i * 4
            chunk = payload[start : start + 4]
            if self.value_type == "float32":
                values.append(float(struct.unpack(float_fmt, chunk)[0]))
            else:
                values.append(float(struct.unpack(int_fmt, chunk)[0]))
        return values

    def _decode_inner_frames(self, payload):
        if len(payload) != self._outer_payload_len:
            return None
        out = []
        header_len = len(self.inner_header)
        tail_len = len(self.inner_tail)
        for i in range(self.outer_frame_count):
            start = i * self._inner_frame_len
            frame = payload[start : start + self._inner_frame_len]
            if frame[:header_len] != self.inner_header:
                return None
            if frame[self._inner_frame_len - tail_len : self._inner_frame_len] != self.inner_tail:
                return None
            out.append(self._decode_payload(frame[header_len : header_len + self._inner_payload_len]))
        return out

    def feed(self, data):
        if not data:
            return []
        self._buf.extend(data)
        out = []
        header_len = len(self.outer_header)
        tail_len = len(self.outer_tail)

        while len(self._buf) >= self._outer_frame_len:
            idx = self._buf.find(self.outer_header)
            if idx < 0:
                keep = max(0, header_len - 1)
                if keep > 0 and len(self._buf) > keep:
                    self._buf = bytearray(self._buf[-keep:])
                elif keep == 0:
                    self._buf = bytearray()
                break
            if idx > 0:
                self._buf = bytearray(self._buf[idx:])
            if len(self._buf) < self._outer_frame_len:
                break

            tail_start = header_len + self._outer_payload_len
            if self._buf[tail_start : tail_start + tail_len] != self.outer_tail:
                self._buf = bytearray(self._buf[1:])
                continue

            payload = self._buf[header_len:tail_start]
            values_list = self._decode_inner_frames(payload)
            if values_list is None:
                self._buf = bytearray(self._buf[1:])
                continue

            out.extend(values_list)
            self._buf = bytearray(self._buf[self._outer_frame_len :])
        return out


class UartFixedLengthParser:
    """
    固定长度分包器。

    用于联调阶段的弱校验测试：
    1. 不检查帧头和帧尾。
    2. 只要累计到固定长度，就切出一包。
    3. 适合确认“对端是否真的把 52 字节完整送到 K230”。
    """

    def __init__(self, frame_len):
        self.frame_len = int(frame_len)
        self._buf = bytearray()

    def feed(self, data):
        if not data:
            return []
        self._buf.extend(data)
        out = []
        while len(self._buf) >= self.frame_len:
            out.append(bytes(self._buf[: self.frame_len]))
            self._buf = bytearray(self._buf[self.frame_len :])
        return out


def file_size_mtime(path):
    # 取文件大小与修改时间，供离线缓存键计算使用。
    try:
        st = os.stat(path)
    except OSError:
        return 0, 0
    size = 0
    mtime = 0
    try:
        size = int(st[6])
    except Exception:
        size = 0
    try:
        mtime = int(st[8])
    except Exception:
        try:
            mtime = int(st[-2])
        except Exception:
            mtime = 0
    return size, mtime


def norm_path(path):
    # 统一路径分隔符，减少 Windows/Unix 风格差异带来的判断分支。
    return str(path).replace("\\", "/")


def join_path(base, rel):
    # 轻量路径拼接函数，适配 MicroPython 环境。
    rel = norm_path(rel)
    if rel.startswith("/"):
        return rel
    base = norm_path(base)
    if base.endswith("/"):
        return base + rel
    return base + "/" + rel


def dirname(path):
    # 轻量 dirname 实现，避免依赖不一致的 os.path。
    p = norm_path(path).rstrip("/")
    idx = p.rfind("/")
    if idx < 0:
        return "."
    if idx == 0:
        return "/"
    return p[:idx]


def exists(path):
    # 通过 stat 判断路径是否存在。
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def ensure_dir(path):
    # 逐级创建目录，兼容板端较简化的文件系统接口。
    p = norm_path(path)
    if p in {"", ".", "/"}:
        return
    abs_path = p.startswith("/")
    cur = "/" if abs_path else ""
    parts = [seg for seg in p.strip("/").split("/") if seg]
    for seg in parts:
        if cur == "/":
            cur = "/" + seg
        elif cur == "":
            cur = seg
        else:
            cur = cur + "/" + seg
        try:
            os.mkdir(cur)
        except OSError:
            pass


def list_csv_files(data_dir):
    # 列出测试目录下的所有 CSV，并按名称排序，保证取样顺序稳定。
    # 中文注释：如果配置直接指向单个 CSV 文件，也允许只读取这一份数据，便于板端快速定向测试。
    if str(data_dir).lower().endswith(".csv"):
        if exists(data_dir):
            return [data_dir]
        return []
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    files = []
    for name in names:
        if str(name).lower().endswith(".csv"):
            files.append(join_path(data_dir, name))
    files.sort()
    return files


def file_stem(path):
    # 取文件主名，不带扩展名。
    name = norm_path(path).split("/")[-1]
    dot = name.rfind(".")
    if dot > 0:
        return name[:dot]
    return name


def load_json(path):
    # 统一 JSON 读取入口。
    with open(path, "r") as f:
        return json.load(f)


def is_abs_path(path):
    # 判断路径是否已经是绝对路径，兼容 Windows 和板端 `/sdcard/...` 写法。
    text = norm_path(path)
    if not text:
        return False
    if text.startswith("/"):
        return True
    return len(text) > 2 and text[1] == ":" and text[2] == "/"


def resolve_runtime_config_path(root, cli_args):
    # 允许通过 `--config xxx.json` 或首个 json/jsonc 位置参数显式指定配置文件。
    # 这样同一套脚本就能配合 `raw_cnn_k230/configs/` 下的多份配置使用。
    selected = None
    args = list(cli_args or [])
    idx = 0
    while idx < len(args):
        token = str(args[idx])
        if token == "--config":
            if idx + 1 >= len(args):
                raise ValueError("--config requires a path argument.")
            selected = str(args[idx + 1])
            break
        if token.lower().endswith(".json") or token.lower().endswith(".jsonc"):
            selected = token
            break
        idx += 1

    if OVERRIDE_CONFIG_PATH:
        selected = str(OVERRIDE_CONFIG_PATH)

    if not selected:
        return join_path(root, DEFAULT_RUNTIME_CONFIG_PATH)
    if is_abs_path(selected):
        return norm_path(selected)
    return join_path(root, selected)


def normalize_runtime_mode(raw_mode):
    # 缁熶竴杩愯妯″紡鍛藉悕锛涙妸鍘嗗彶鍒悕鏀跺彛涓烘爣鍑嗗啓娉曘€?
    mode = str(raw_mode or "csv_cached").strip().lower()
    alias_map = {
        "online_uart": "uart_online",
        "echo": "uart_echo",
        "frame_return": "uart_frame_return",
        "debug_ack": "uart_debug_ack",
        "ack": "uart_debug_ack",
    }
    return alias_map.get(mode, mode)


def get_runtime_section(runtime_cfg, section_name):
    # 浼樺厛璇昏鑼冮厤缃潡锛屽叧閿椂鍏煎鏃у瓧娈靛悕銆?
    if not isinstance(runtime_cfg, dict):
        return {}
    section = runtime_cfg.get(section_name, None)
    if isinstance(section, dict):
        return section
    legacy_map = {
        "uart_online": "online_uart",
    }
    legacy_name = legacy_map.get(section_name, None)
    legacy_section = runtime_cfg.get(legacy_name, None)
    if isinstance(legacy_section, dict):
        return legacy_section
    return {}


def require_positive_int(value, field_name):
    # 对配置项做正整数检查，发现异常立即报错。
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(field_name + " must be > 0, got " + str(parsed))
    return parsed


def resolve_positive_step(value, fallback, field_name):
    # step 类配置允许为空；为空时使用 fallback。
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


def parse_label_from_name(filename):
    # 离线评估时，从文件名里解析真实标签。
    # 例如 0.123-xx.csv -> 0.123
    stem = file_stem(filename)
    if "-" not in stem:
        return float("nan")
    token = stem.split("-")[0]
    try:
        return float(token)
    except ValueError:
        return float("nan")


def read_signal(csv_path):
    # 当前离线 CSV 只读取每行第 1 列作为一条原始信号。
    values = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",")[0].strip()
            try:
                values.append(float(first))
            except ValueError:
                continue
    return as_float_array(values)


def finalize_dataset(X_list, y_list, seq_length):
    # 把 Python 列表整理成模型可直接使用的三维数组。
    if not X_list:
        return empty_float((0, seq_length, 0)), empty_float((0,))

    sample_width = int(X_list[0].shape[1]) if len(X_list) > 0 else 0
    X = empty_float((len(X_list), seq_length, sample_width))
    for i in range(len(X_list)):
        X[i] = X_list[i]
    return astype_float_array(X), as_float_array(y_list)


def normalize_feature_mode(feature_mode):
    text = str(feature_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"window_demean", "demean", "window_mean_center"}:
        return "window_demean"
    if text in {"window_rel_demean", "relative_demean", "window_mean_ratio"}:
        return "window_rel_demean"
    return "raw"


def normalize_model_type(model_type):
    # 统一模型类型写法，兼容 cnn / cnn_all / cnn_lstm 等别名。
    text = str(model_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"cnn", "cnn_all", "raw_cnn"}:
        return "cnn"
    if text in {"cnn_lstm", "cnnlstm"}:
        return "cnn_lstm"
    if text in {"cnn_tcn", "cnntcn"}:
        return "cnn_tcn"
    if text in {"cnn_tcn_seg3_soft_stats_moe", "cnn_tcn_seg3", "cnntcnseg3"}:
        return "cnn_tcn_seg3_soft_stats_moe"
    return ""


def get_model_type(cfg):
    # 优先读取配置中的 model.type；若缺省，则根据 sequence_length 自动推断。
    model_cfg = cfg.get("model", {})
    text = normalize_model_type(model_cfg.get("type", ""))
    if text:
        return text
    data_cfg = cfg.get("data", {})
    seq_length = require_positive_int(data_cfg.get("sequence_length", 1), "data.sequence_length")
    if seq_length <= 1:
        return "cnn"
    return "cnn_lstm"


def get_feature_mode(cfg):
    preprocessing_cfg = cfg.get("preprocessing", {})
    return normalize_feature_mode(preprocessing_cfg.get("feature_mode", "raw"))


def apply_feature_mode_1d(src_window, feature_mode, out_window):
    mode = normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        mean_value = float(np.sum(src_window) / float(len(src_window)))
        out_window[:] = src_window - mean_value
        return out_window
    if mode == "window_rel_demean":
        # 中文注释：先去窗口均值，再除以均值绝对值，避免不同基线频率直接主导幅度。
        mean_value = float(np.sum(src_window) / float(len(src_window)))
        denom = abs(mean_value)
        if denom < 1e-6:
            denom = 1e-6
        out_window[:] = (src_window - mean_value) / denom
        return out_window
    out_window[:] = src_window
    return out_window


def build_dataset(cfg, root, max_samples=None):
    """
    离线模式的数据构建函数。

    流程：
    1. 遍历 test_data 下的 CSV。
    2. 对每个 CSV 用滑动窗口切成多个片段。
    3. 按 sequence_length 再组织成模型输入样本。
    4. 收集样本 X 与标签 y。
    """
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    data_dir = join_path(root, paths["test_data_dir"])

    base_window = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step_cfg = data_cfg.get("base_step", None)
    base_step = resolve_positive_step(base_step_cfg, base_window // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")
    feature_mode = get_feature_mode(cfg)
    if max_samples is not None:
        max_samples = require_positive_int(max_samples, "runtime.max_samples")

    X_list = []
    y_list = []
    for csv_file in list_csv_files(data_dir):
        print("dataset_read_csv:", csv_file)
        signal = read_signal(csv_file)
        if signal.size < base_window:
            continue
        # features 先保存所有基础窗口，再按 sequence_length 拼装为样本。
        features = []
        next_emit_start = 0
        for start in range(0, signal.size - base_window + 1, base_step):
            window = astype_float_array(signal[start : start + base_window])
            proc_window = empty_float((base_window,))
            apply_feature_mode_1d(window, feature_mode, proc_window)
            features.append(proc_window)
            # 一边切窗一边吐样本，便于 max_samples 提前截断，减少无意义构建。
            while next_emit_start + seq_length <= len(features):
                sample = empty_float((seq_length, base_window))
                seg = features[next_emit_start : next_emit_start + seq_length]
                for j in range(seq_length):
                    sample[j] = seg[j]
                X_list.append(sample)
                y_list.append(parse_label_from_name(csv_file))
                next_emit_start += seq_step
                if max_samples is not None and len(X_list) >= max_samples:
                    print("dataset_limit_reached:", len(X_list))
                    return finalize_dataset(X_list, y_list, seq_length)
                if len(X_list) % 100 == 0:
                    print("dataset_samples_collected:", len(X_list))

    return finalize_dataset(X_list, y_list, seq_length)


def scale_features(X, scaler_json_path):
    # 读取训练阶段导出的 mean / scale，对离线样本做同样的标准化。
    scaler = load_json(scaler_json_path)
    mean = as_float_array(scaler["mean"])
    scale = as_float_array(scaler["scale"])
    eps = 1e-12
    for i in range(len(scale)):
        if abs(float(scale[i])) < eps:
            scale[i] = 1.0
    X_flat = astype_float_array(X.reshape((X.shape[0] * X.shape[1], X.shape[-1])))
    X_scaled = (X_flat - mean) / scale
    return astype_float_array(X_scaled.reshape(X.shape))


def load_scaler_params(scaler_json_path):
    # 在线模式不需要提前构建整批数据，只需单独拿到标准化参数即可。
    scaler = load_json(scaler_json_path)
    mean = as_float_array(scaler["mean"])
    scale = as_float_array(scaler["scale"])
    eps = 1e-12
    for i in range(len(scale)):
        if abs(float(scale[i])) < eps:
            scale[i] = 1.0
    return mean, scale


def expand_ring_window(ring_row, write_idx, out_window):
    # 将某一路环形缓冲展开为连续窗口，顺序为“最旧 -> 最新”。
    n = int(len(ring_row))
    idx = int(write_idx) % n
    if idx == 0:
        out_window[:] = ring_row
        return out_window
    right = n - idx
    out_window[:right] = ring_row[idx:]
    out_window[right:] = ring_row[:idx]
    return out_window


def expand_sequence_ring(seq_ring, write_idx, out_seq):
    # 灏嗗簭鍒楃幆褰㈢紦鍐插睍寮€涓衡€滄渶鏃?-> 鏈€鏂扳€濈殑鍏ㄥ簭鍒椼€?
    n = int(len(seq_ring))
    idx = int(write_idx) % n
    if idx == 0:
        out_seq[:] = seq_ring
        return out_seq
    right = n - idx
    out_seq[:right] = seq_ring[idx:]
    out_seq[right:] = seq_ring[:idx]
    return out_seq


ZERO_GUARD_DEFAULT_THRESHOLDS = {
    "diff_p95_abs": 75.0,
    "win_range_mean": 280.0,
    "win_std_mean": 40.0,
    "absz_mean": 0.012,
}

ZERO_GUARD_DEFAULT_CONFIG = {
    "enabled": False,
    "output_value": 0.0,
    "freq_enter_threshold": 480000.0,
    "freq_exit_threshold": 500000.0,
    "enter_consecutive_windows": 3,
    "exit_consecutive_windows": 3,
    "confidence_absz_threshold": 3.0,
}


def get_zero_guard_config(cfg):
    # 中文注释：板端优先读取顶层 zero_guard；兼容把配置写在 preprocessing.zero_guard 下面的形式。
    guard_cfg = cfg.get("zero_guard", None)
    if guard_cfg is None:
        preprocessing_cfg = cfg.get("preprocessing", {})
        guard_cfg = preprocessing_cfg.get("zero_guard", {})
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}
    return guard_cfg


def read_zero_guard_runtime_config(guard_cfg):
    # 中文注释：新判据只使用频率阈值和连续窗口数；旧 thresholds/min_votes 仅保留兼容读取。
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}
    runtime_cfg = dict(ZERO_GUARD_DEFAULT_CONFIG)
    for key in ZERO_GUARD_DEFAULT_CONFIG:
        if guard_cfg.get(key, None) is not None:
            if key == "enabled":
                runtime_cfg[key] = bool(guard_cfg[key])
            elif key in ("enter_consecutive_windows", "exit_consecutive_windows"):
                value = int(guard_cfg[key])
                if value <= 0:
                    value = 1
                runtime_cfg[key] = value
            else:
                runtime_cfg[key] = float(guard_cfg[key])
    return runtime_cfg


def read_zero_guard_thresholds(guard_cfg):
    # 中文注释：复制默认阈值，再叠加用户配置，避免缺字段时板端启动失败。
    thresholds = dict(ZERO_GUARD_DEFAULT_THRESHOLDS)
    user_thresholds = guard_cfg.get("thresholds", {})
    if isinstance(user_thresholds, dict):
        for key in ZERO_GUARD_DEFAULT_THRESHOLDS:
            if user_thresholds.get(key, None) is not None:
                thresholds[key] = float(user_thresholds[key])
    for key in ZERO_GUARD_DEFAULT_THRESHOLDS:
        if guard_cfg.get(key, None) is not None:
            thresholds[key] = float(guard_cfg[key])
    return thresholds


def zero_guard_percentile_from_sorted(values, pct):
    # 中文注释：MicroPython/ulab 上不依赖 np.percentile，直接用排序后的近似分位点。
    n = int(len(values))
    if n <= 0:
        return 0.0
    values.sort()
    idx = int((float(n) - 1.0) * float(pct) / 100.0 + 0.5)
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return float(values[idx])


def zero_guard_mean_all(values):
    # 中文注释：兼容 1D/2D/3D 数组，按所有原始频率点展平求均值。
    shape = values.shape
    ndim = int(len(shape))
    total = 0.0
    count = 0
    if ndim == 1:
        for i in range(int(shape[0])):
            total += float(values[i])
            count += 1
    elif ndim == 2:
        for i in range(int(shape[0])):
            for j in range(int(shape[1])):
                total += float(values[i][j])
                count += 1
    else:
        for i in range(int(shape[0])):
            for j in range(int(shape[1])):
                for k in range(int(shape[2])):
                    total += float(values[i][j][k])
                    count += 1
    if count <= 0:
        return 0.0
    return total / float(count)


def compute_zero_guard_features(raw_seq, scaled_seq):
    # 中文注释：raw_seq 是未做 window_demean 的原始窗口序列，scaled_seq 是送入模型前的标准化序列。
    guard_cfg = read_zero_guard_runtime_config({})
    raw_shape = raw_seq.shape
    raw_ndim = int(len(raw_shape))
    if raw_ndim <= 1:
        seq_len = 1
        width = int(raw_shape[0])
    else:
        seq_len = int(raw_shape[0])
        width = int(raw_shape[1])
    if seq_len <= 0 or width <= 0:
        return {
            "win_std_mean": 0.0,
            "win_range_mean": 0.0,
            "diff_p95_abs": 0.0,
            "freq_mean": 0.0,
            "absz_mean": 0.0,
            "zero_identity": False,
            "zero_confidence_high": False,
            "zero_guard_state": False,
        }

    std_total = 0.0
    range_total = 0.0
    diff_abs = []
    for t in range(seq_len):
        if raw_ndim <= 1:
            row = raw_seq
        else:
            row = raw_seq[t]
        row_min = float(row[0])
        row_max = float(row[0])
        row_sum = 0.0
        for i in range(width):
            v = float(row[i])
            row_sum += v
            if v < row_min:
                row_min = v
            if v > row_max:
                row_max = v
            if i > 0:
                d = v - float(row[i - 1])
                if d < 0.0:
                    d = -d
                diff_abs.append(d)
        row_mean = row_sum / float(width)
        var_sum = 0.0
        for i in range(width):
            d = float(row[i]) - row_mean
            var_sum += d * d
        std_total += float(np.sqrt(var_sum / float(width)))
        range_total += row_max - row_min

    absz_total = 0.0
    absz_count = 0
    if scaled_seq is not None:
        for t in range(int(scaled_seq.shape[0])):
            for i in range(int(scaled_seq.shape[1])):
                v = float(scaled_seq[t][i])
                if v < 0.0:
                    v = -v
                absz_total += v
                absz_count += 1

    absz_mean = 0.0
    if absz_count > 0:
        absz_mean = absz_total / float(absz_count)

    freq_mean = zero_guard_mean_all(raw_seq)
    return {
        "win_std_mean": std_total / float(seq_len),
        "win_range_mean": range_total / float(seq_len),
        "diff_p95_abs": zero_guard_percentile_from_sorted(diff_abs, 95.0),
        "freq_mean": freq_mean,
        "absz_mean": absz_mean,
        "zero_identity": freq_mean <= float(guard_cfg["freq_enter_threshold"]),
        "zero_confidence_high": absz_mean >= float(guard_cfg["confidence_absz_threshold"]),
        "zero_guard_state": False,
    }


class ZeroGuardState:
    # 中文注释：按推理窗口顺序维护 0 干度保护的进入/退出滞回状态。
    def __init__(self, guard_cfg=None):
        self.configure(guard_cfg)

    def configure(self, guard_cfg=None):
        cfg = read_zero_guard_runtime_config(guard_cfg or {})
        self.freq_enter_threshold = float(cfg["freq_enter_threshold"])
        self.freq_exit_threshold = float(cfg["freq_exit_threshold"])
        self.enter_consecutive_windows = int(cfg["enter_consecutive_windows"])
        self.exit_consecutive_windows = int(cfg["exit_consecutive_windows"])
        self.confidence_absz_threshold = float(cfg["confidence_absz_threshold"])
        self.active = False
        self.enter_count = 0
        self.exit_count = 0

    def update(self, freq_mean):
        freq_mean = float(freq_mean)
        if self.active:
            self.enter_count = 0
            if freq_mean >= self.freq_exit_threshold:
                self.exit_count += 1
            else:
                self.exit_count = 0
            if self.exit_count >= self.exit_consecutive_windows:
                self.active = False
                self.enter_count = 0
                self.exit_count = 0
        else:
            self.exit_count = 0
            if freq_mean <= self.freq_enter_threshold:
                self.enter_count += 1
            else:
                self.enter_count = 0
            if self.enter_count >= self.enter_consecutive_windows:
                self.active = True
                self.exit_count = 0
        return bool(self.active)


def compute_zero_guard_stateful_mask(raw_sequences, scaled_sequences=None, config=None):
    # 中文注释：离线按样本顺序扫描，每个样本可为 (F,) 或 (T, F)，返回连续窗口保护状态。
    guard_cfg = config or {}
    runtime_cfg = read_zero_guard_runtime_config(guard_cfg)
    raw_shape = raw_sequences.shape
    raw_ndim = int(len(raw_shape))
    if raw_ndim <= 1:
        sample_count = 1
    else:
        sample_count = int(raw_shape[0])

    state = ZeroGuardState(guard_cfg)
    mask = []
    votes = []
    feature_rows = {
        "freq_mean": [],
        "absz_mean": [],
        "diff_p95_abs": [],
        "win_range_mean": [],
        "win_std_mean": [],
        "zero_identity": [],
        "zero_confidence_high": [],
        "zero_guard_state": [],
        "enter_count": [],
        "exit_count": [],
    }
    for i in range(sample_count):
        if raw_ndim <= 1:
            raw_sample = raw_sequences
        else:
            raw_sample = raw_sequences[i]
        scaled_sample = None
        if scaled_sequences is not None:
            scaled_shape = scaled_sequences.shape
            if int(len(scaled_shape)) <= 1:
                scaled_sample = scaled_sequences
            else:
                scaled_sample = scaled_sequences[i]

        features = compute_zero_guard_features(raw_sample, scaled_sample)
        freq_mean = float(features.get("freq_mean", 0.0))
        absz_mean = float(features.get("absz_mean", 0.0))
        features["zero_identity"] = freq_mean <= float(runtime_cfg["freq_enter_threshold"])
        features["zero_confidence_high"] = absz_mean >= float(runtime_cfg["confidence_absz_threshold"])
        if bool(runtime_cfg["enabled"]):
            state.update(freq_mean)
        features["zero_guard_state"] = bool(state.active)
        features["enter_count"] = int(state.enter_count)
        features["exit_count"] = int(state.exit_count)

        mask.append(bool(state.active))
        votes.append(int(state.enter_count))
        for key in feature_rows:
            feature_rows[key].append(features.get(key, 0.0))
    return mask, feature_rows, votes


def is_zero_guard_hit(raw_seq, scaled_seq, guard_cfg, state=None):
    # 中文注释：0 干度身份只由原始频率均值判定，低波动特征仅作为诊断日志。
    if not bool(guard_cfg.get("enabled", False)):
        return False, 0, {}
    runtime_cfg = read_zero_guard_runtime_config(guard_cfg)
    features = compute_zero_guard_features(raw_seq, scaled_seq)
    freq_mean = float(features.get("freq_mean", 0.0))
    absz_mean = float(features.get("absz_mean", 0.0))
    features["zero_identity"] = freq_mean <= float(runtime_cfg["freq_enter_threshold"])
    features["zero_confidence_high"] = absz_mean >= float(runtime_cfg["confidence_absz_threshold"])
    if state is None:
        state = ZeroGuardState(guard_cfg)
        shape = raw_seq.shape
        if int(len(shape)) >= 2:
            for t in range(int(shape[0])):
                state.update(zero_guard_mean_all(raw_seq[t]))
        else:
            state.update(freq_mean)
    else:
        state.update(freq_mean)
    features["zero_guard_state"] = bool(state.active)
    features["enter_count"] = int(state.enter_count)
    features["exit_count"] = int(state.exit_count)
    return bool(state.active), int(state.enter_count), features


def get_postprocessing_config(cfg):
    # 中文注释：后处理只作用于模型预测值，不改变 kmodel 输入输出，也不参与 zero_guard 判定。
    pp_cfg = cfg.get("postprocessing", {})
    if not isinstance(pp_cfg, dict):
        pp_cfg = {}
    return pp_cfg


def normalize_postprocessing_type(raw_type):
    text = str(raw_type or "None").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text in {"", "none", "off", "disable", "disabled", "false"}:
        return "none"
    if text in {"exponential", "exponential_smoothing", "exponential_smoother", "exp"}:
        return "exponential"
    if text in {"kalman", "kalman_smoother", "kalman_filter"}:
        return "kalman"
    raise ValueError("Unsupported postprocessing.type: " + str(raw_type))


class RuntimePostprocessor:
    # 中文注释：K230 在线推理是流式输出，这里为每个通道单独保存平滑状态。
    def __init__(self, cfg, channel_count=1):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.kind = normalize_postprocessing_type(cfg.get("type", "None"))
        if self.kind == "none":
            self.enabled = False
        self.channel_count = int(channel_count)
        if self.channel_count <= 0:
            self.channel_count = 1
        self.alpha = float(cfg.get("exp_smooth_alpha", 0.3))
        if self.alpha < 0.0:
            self.alpha = 0.0
        if self.alpha > 1.0:
            self.alpha = 1.0
        self.kalman_q = float(cfg.get("kalman_q", 0.001))
        self.kalman_r = float(cfg.get("kalman_r", 0.1))
        if self.kalman_q < 0.0:
            self.kalman_q = 0.0
        if self.kalman_r <= 0.0:
            self.kalman_r = 0.1
        self.apply_to_zero_guard = bool(cfg.get("apply_to_zero_guard", False))
        self.reset_on_zero_guard = bool(cfg.get("reset_on_zero_guard", True))
        self._exp_values = [None] * self.channel_count
        self._kalman_x = [None] * self.channel_count
        self._kalman_p = [1.0] * self.channel_count

    def reset_channel(self, channel):
        idx = int(channel)
        if idx < 0 or idx >= self.channel_count:
            return
        self._exp_values[idx] = None
        self._kalman_x[idx] = None
        self._kalman_p[idx] = 1.0

    def reset_all(self):
        for i in range(self.channel_count):
            self.reset_channel(i)

    def update(self, channel, value, zero_guard_hit=False):
        if not self.enabled:
            return float(value)
        idx = int(channel)
        if idx < 0:
            idx = 0
        if idx >= self.channel_count:
            idx = self.channel_count - 1
        if zero_guard_hit and not self.apply_to_zero_guard:
            if self.reset_on_zero_guard:
                self.reset_channel(idx)
            return float(value)
        if self.kind == "exponential":
            return self._update_exponential(idx, value)
        if self.kind == "kalman":
            return self._update_kalman(idx, value)
        return float(value)

    def _update_exponential(self, channel, value):
        measurement = float(value)
        last = self._exp_values[channel]
        if last is None:
            self._exp_values[channel] = measurement
            return measurement
        smoothed = self.alpha * measurement + (1.0 - self.alpha) * float(last)
        self._exp_values[channel] = smoothed
        return smoothed

    def _update_kalman(self, channel, value):
        measurement = float(value)
        x = self._kalman_x[channel]
        if x is None:
            self._kalman_x[channel] = measurement
            self._kalman_p[channel] = 1.0
            return measurement
        p_pred = float(self._kalman_p[channel]) + self.kalman_q
        k_gain = p_pred / (p_pred + self.kalman_r)
        x_new = float(x) + k_gain * (measurement - float(x))
        self._kalman_x[channel] = x_new
        self._kalman_p[channel] = (1.0 - k_gain) * p_pred
        return x_new


def create_runtime_postprocessor(cfg, channel_count=1):
    pp_cfg = get_postprocessing_config(cfg)
    return RuntimePostprocessor(pp_cfg, channel_count=channel_count)


def mean_1d(values):
    # 中文注释：报警判断要用原始窗口均值，手写循环避免板端额外依赖。
    count = int(len(values))
    if count <= 0:
        return 0.0
    total = 0.0
    for i in range(count):
        total += float(values[i])
    return total / float(count)


def median_list(values):
    # 中文注释：多路或多模型预测用中位数融合，降低单点跳动对报警的影响。
    count = int(len(values))
    if count <= 0:
        return 0.0
    ordered = sorted([float(v) for v in values])
    mid = count // 2
    if (count % 2) == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def clamp_count(value, fallback, minimum, maximum):
    try:
        out = int(value)
    except Exception:
        out = int(fallback)
    if out < minimum:
        out = minimum
    if out > maximum:
        out = maximum
    return out


class FullGasAlarmState:
    """
    中文注释：通用满气报警状态机。

    这个状态机不关心当前是单模型还是多模型，只接收：
    1. 本轮预测值列表。
    2. 本轮原始平均频率。
    3. 本轮是否命中 zero_guard。
    """

    def __init__(self, cfg):
        if not isinstance(cfg, dict):
            cfg = {}
        self.enabled = bool(cfg.get("enabled", False))
        self.output_name = str(cfg.get("output_name", "full_gas_alarm"))
        self.output_slot = int(cfg.get("output_slot", 3))
        self.on_value = float(cfg.get("on_value", 1.0))
        self.off_value = float(cfg.get("off_value", 0.0))

        self.history_size = clamp_count(cfg.get("history_size", 6), 6, 2, 64)
        self.recent_count = clamp_count(cfg.get("recent_count", 3), 3, 1, self.history_size)

        self.danger_threshold = float(cfg.get("danger_threshold", 0.55))
        self.danger_min_count = clamp_count(cfg.get("danger_min_count", 5), 5, 1, self.history_size)
        self.alarm_threshold = float(cfg.get("alarm_threshold", cfg.get("threshold_on", 0.60)))
        self.alarm_recent_count = clamp_count(
            cfg.get("alarm_recent_count", self.recent_count),
            self.recent_count,
            1,
            self.history_size,
        )
        self.alarm_recent_min_count = clamp_count(
            cfg.get("alarm_recent_min_count", 2),
            2,
            1,
            self.alarm_recent_count,
        )

        self.threshold_off = float(cfg.get("threshold_off", 0.52))
        self.mean_off = float(cfg.get("mean_off", 0.54))
        self.recent_soft_off = float(cfg.get("recent_soft_off", 0.58))
        self.min_low_count = clamp_count(cfg.get("min_low_count", 4), 4, 1, self.history_size)

        self.freq_slope_min = float(cfg.get("freq_slope_min", 1.0))
        self.freq_rise_min_count = clamp_count(cfg.get("freq_rise_min_count", 4), 4, 1, self.history_size - 1)
        self.freq_stop_delta = float(cfg.get("freq_stop_delta", 0.0))
        self.hit_on_count = clamp_count(cfg.get("hit_on_count", 2), 2, 1, 64)
        self.hit_off_count = clamp_count(cfg.get("hit_off_count", 3), 3, 1, 64)

        self.dry_history = []
        self.freq_history = []
        self.alarm_on = False
        self.on_hits = 0
        self.off_hits = 0
        self.last_reason = "disabled" if not self.enabled else "warming"

    def _append_history(self, history, value):
        history.append(float(value))
        while len(history) > self.history_size:
            del history[0]

    def _mean_recent(self, history, count):
        usable = int(count)
        if usable <= 0:
            return 0.0
        if usable > len(history):
            usable = len(history)
        if usable <= 0:
            return 0.0
        start = len(history) - usable
        total = 0.0
        for i in range(start, len(history)):
            total += float(history[i])
        return total / float(usable)

    def _mean_all(self, history):
        return self._mean_recent(history, len(history))

    def _count_ge(self, history, threshold):
        count = 0
        for value in history:
            if float(value) >= float(threshold):
                count += 1
        return count

    def _count_le(self, history, threshold):
        count = 0
        for value in history:
            if float(value) <= float(threshold):
                count += 1
        return count

    def _count_recent_ge(self, history, count, threshold):
        usable = int(count)
        if usable > len(history):
            usable = len(history)
        if usable <= 0:
            return 0
        start = len(history) - usable
        out = 0
        for i in range(start, len(history)):
            if float(history[i]) >= float(threshold):
                out += 1
        return out

    def _rise_count(self, history):
        count = 0
        for i in range(1, len(history)):
            if float(history[i]) > float(history[i - 1]):
                count += 1
        return count

    def _linear_slope(self, history):
        count = int(len(history))
        if count <= 1:
            return 0.0
        x_mean = float(count - 1) / 2.0
        y_total = 0.0
        for value in history:
            y_total += float(value)
        y_mean = y_total / float(count)
        num = 0.0
        den = 0.0
        for i in range(count):
            dx = float(i) - x_mean
            num += dx * (float(history[i]) - y_mean)
            den += dx * dx
        if den <= 0.0:
            return 0.0
        return num / den

    def update(self, model_values, freq_mean, zero_guard_hit=False):
        if not self.enabled:
            return self.off_value

        dry_value = median_list(model_values)
        self._append_history(self.dry_history, dry_value)
        self._append_history(self.freq_history, freq_mean)

        if len(self.dry_history) < self.history_size or len(self.freq_history) < self.history_size:
            self.last_reason = "warming"
            return self.on_value if self.alarm_on else self.off_value

        dry_mean = self._mean_all(self.dry_history)
        dry_recent_mean = self._mean_recent(self.dry_history, self.recent_count)
        danger_count = self._count_ge(self.dry_history, self.danger_threshold)
        alarm_recent_count = self._count_recent_ge(
            self.dry_history,
            self.alarm_recent_count,
            self.alarm_threshold,
        )
        low_count = self._count_le(self.dry_history, self.threshold_off)
        freq_delta = float(self.freq_history[-1]) - float(self.freq_history[0])
        freq_slope = self._linear_slope(self.freq_history)
        freq_rise_count = self._rise_count(self.freq_history)

        dry_high = danger_count >= self.danger_min_count
        alarm_line_hit = alarm_recent_count >= self.alarm_recent_min_count
        freq_rising = (
            freq_slope >= self.freq_slope_min
            and freq_rise_count >= self.freq_rise_min_count
        )
        alarm_condition = dry_high and alarm_line_hit and freq_rising and not bool(zero_guard_hit)

        clear_by_low = dry_mean <= self.mean_off or low_count >= self.min_low_count
        clear_by_trend_stop = dry_recent_mean <= self.recent_soft_off and freq_delta <= self.freq_stop_delta
        clear_condition = clear_by_low or clear_by_trend_stop or bool(zero_guard_hit)

        if self.alarm_on:
            if clear_condition:
                self.off_hits += 1
            else:
                self.off_hits = 0
            self.on_hits = 0
            if self.off_hits >= self.hit_off_count:
                self.alarm_on = False
                self.off_hits = 0
                self.last_reason = "clear"
            else:
                self.last_reason = "hold_on"
        else:
            if alarm_condition:
                self.on_hits += 1
            else:
                self.on_hits = 0
            self.off_hits = 0
            if self.on_hits >= self.hit_on_count:
                self.alarm_on = True
                self.on_hits = 0
                self.last_reason = "alarm"
            else:
                self.last_reason = "hold_off"

        return self.on_value if self.alarm_on else self.off_value

    def summary(self):
        if not self.enabled:
            return "disabled"
        return (
            "enabled=True, code=0x{:02X}, history_size={}, "
            "danger_threshold={}, alarm_threshold={}, threshold_off={}, state={}"
        ).format(
            FULL_GAS_ALARM_CODE,
            self.history_size,
            self.danger_threshold,
            self.alarm_threshold,
            self.threshold_off,
            bool(self.alarm_on),
        )


def update_full_gas_alarm_state(alarm_state, model_values, freq_mean, zero_guard_hit=False):
    # 中文注释：只更新高干度报警状态；报警结果通过返回码表达，不再改写干度输出值。
    if alarm_state is None or not alarm_state.enabled:
        return None
    return alarm_state.update(model_values, freq_mean, zero_guard_hit=zero_guard_hit)


def make_dataset_cache_key(cfg, root, max_samples, scaler_json_path):
    # 用配置项与 CSV/Scaler 文件元数据生成缓存键。
    # 只要关键输入变化，缓存键就会变化，从而触发重建。
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    data_dir = join_path(root, paths["test_data_dir"])
    files = list_csv_files(data_dir)

    parts = [
        norm_path(root),
        str(max_samples),
        str(get_feature_mode(cfg)),
        str(data_cfg.get("base_window_size")),
        str(data_cfg.get("base_step")),
        str(data_cfg.get("sequence_length")),
        str(data_cfg.get("sequence_step")),
    ]
    scaler_size, scaler_mtime = file_size_mtime(scaler_json_path)
    parts.append("scaler:{}:{}".format(scaler_size, scaler_mtime))
    for p in files:
        size, mtime = file_size_mtime(p)
        parts.append("{}:{}:{}".format(norm_path(p), size, mtime))
    return "|".join(parts)


def ensure_dataset_cache(cfg, root, max_samples, scaler_json_path):
    # 离线模式入口：
    # 若缓存命中，直接复用已标准化的数据；
    # 若缓存失效，再重新遍历 CSV 构建。
    cache_key = make_dataset_cache_key(cfg, root, max_samples, scaler_json_path)
    model_type = get_model_type(cfg)
    cached_X = RUNTIME_CACHE.get("X_scaled", None)
    cached_raw = RUNTIME_CACHE.get("X_raw_aux", None)
    cached_y = RUNTIME_CACHE.get("y", None)
    if (
        RUNTIME_CACHE.get("dataset_key", None) == cache_key
        and cached_X is not None
        and cached_y is not None
        and int(cached_X.shape[0]) > 0
        and (model_type != "cnn_tcn_seg3_soft_stats_moe" or cached_raw is not None)
    ):
        return cached_X, cached_y, False

    print("dataset_cache_miss_rebuild")
    X, y = build_dataset(cfg, root, max_samples=max_samples)
    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found in test_data.")
    X_raw_aux = None
    if model_type == "cnn_tcn_seg3_soft_stats_moe":
        X_raw_aux = astype_float_array(X.copy())
    X_scaled = scale_features(X, scaler_json_path)
    del X
    gc.collect()

    RUNTIME_CACHE["dataset_key"] = cache_key
    RUNTIME_CACHE["X_scaled"] = X_scaled
    RUNTIME_CACHE["X_raw_aux"] = X_raw_aux
    RUNTIME_CACHE["y"] = y
    RUNTIME_CACHE["cursor"] = 0
    return X_scaled, y, True


def make_kmodel_cache_key(kmodel_path):
    # 用模型文件路径/大小/修改时间生成模型缓存键。
    size, mtime = file_size_mtime(kmodel_path)
    return "{}:{}:{}".format(norm_path(kmodel_path), size, mtime)


def ensure_kpu_cache(kmodel_path):
    # 确保 kmodel 只加载一次，后续循环复用同一个 KPU 实例。
    cache_key = make_kmodel_cache_key(kmodel_path)
    if (
        RUNTIME_CACHE.get("kmodel_key", None) == cache_key
        and RUNTIME_CACHE.get("kpu", None) is not None
        and RUNTIME_CACHE.get("nn", None) is not None
    ):
        return RUNTIME_CACHE["nn"], RUNTIME_CACHE["kpu"], False

    import nncase_runtime as nn  # type: ignore

    kpu = nn.kpu()
    kpu.load_kmodel(kmodel_path)
    RUNTIME_CACHE["kmodel_key"] = cache_key
    RUNTIME_CACHE["kpu"] = kpu
    RUNTIME_CACHE["nn"] = nn
    return nn, kpu, True


def acquire_infer_range(total_samples, request_count):
    # 离线缓存模式下，按游标方式取“下一批”样本做推理。
    if total_samples <= 0:
        raise RuntimeError("No cached samples available.")
    count = int(request_count)
    if count <= 0:
        count = 1
    if count > total_samples:
        count = total_samples

    start_idx = int(RUNTIME_CACHE.get("cursor", 0)) % int(total_samples)
    next_cursor = start_idx + count
    while next_cursor >= total_samples:
        next_cursor -= total_samples
    RUNTIME_CACHE["cursor"] = next_cursor
    return start_idx, count


def collect_labels_range(y_all, start_idx, count):
    # 取出与当前推理批次对应的标签，供 MAE/RMSE 统计使用。
    total = int(len(y_all))
    out = empty_float((count,))
    idx = int(start_idx)
    for i in range(count):
        out[i] = y_all[idx]
        idx += 1
        if idx >= total:
            idx = 0
    return out


def run_kmodel_inference_cached(kmodel_path, X_scaled, start_idx, count, uart_sender=None, X_raw_aux=None, postprocessor=None):
    # 在已缓存的离线样本上做一小批推理。
    # 这是当前离线调试模式提速的关键：不再每轮重建全部样本。
    nn, kpu, model_reloaded = ensure_kpu_cache(kmodel_path)
    total = int(X_scaled.shape[0])
    preds = []
    infer_us_total = 0
    idx = int(start_idx)
    for i in range(count):
        sample = astype_float_array(X_scaled[idx])
        sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
        input_tensor = nn.from_numpy(sample)
        kpu.set_input_tensor(0, input_tensor)
        raw_input_tensor = None
        if X_raw_aux is not None:
            raw_sample = astype_float_array(X_raw_aux[idx])
            raw_sample = raw_sample.reshape((1, raw_sample.shape[0], raw_sample.shape[1]))
            raw_input_tensor = nn.from_numpy(raw_sample)
            kpu.set_input_tensor(1, raw_input_tensor)
        t0 = now_us()
        kpu.run()
        t1 = now_us()
        infer_us_total += diff_us(t1, t0)
        output = kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        if postprocessor is not None:
            pred = postprocessor.update(0, pred)
        preds.append(pred)
        if uart_sender is not None:
            uart_sender.send_scaled_prediction(pred)
        del output
        del input_tensor
        if raw_input_tensor is not None:
            del raw_input_tensor
        idx += 1
        if idx >= total:
            idx = 0
        if (i + 1) % 64 == 0:
            gc.collect()
    return as_float_array(preds), infer_us_total, model_reloaded


def run_online_uart_inference_cnn(cfg, root, uart_sender, kmodel_path, scaler_json_path):
    """
    在线串口推理模式。

    目标流程：
    1. 从单片机接收一帧 12 路输入。
    2. 写入 12 路环形缓冲。
    3. 当每路都积累满一个窗口后，逐路推理。
    4. 把 12 路预测结果再按同样协议打包发回去。
    """
    runtime_cfg = cfg.get("runtime", {})
    online_cfg = get_runtime_section(runtime_cfg, "uart_online")
    alarm_cfg = get_runtime_section(runtime_cfg, "full_gas_alarm")
    data_cfg = cfg.get("data", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart_online mode cannot start.")

    window_size = require_positive_int(data_cfg.get("base_window_size", 500), "data.base_window_size")
    seq_length = require_positive_int(data_cfg.get("sequence_length", 1), "data.sequence_length")
    if seq_length != 1:
        raise RuntimeError("uart_online mode currently requires data.sequence_length = 1.")

    channel_count = require_positive_int(online_cfg.get("channel_count", uart_sender.value_count), "runtime.uart_online.channel_count")
    raw_anomaly = RawChannelAnomalyState(get_raw_anomaly_config(cfg), channel_count)
    postprocessor = create_runtime_postprocessor(cfg, channel_count=channel_count)
    full_gas_alarm = FullGasAlarmState(alarm_cfg)
    infer_step_frames = require_positive_int(online_cfg.get("infer_step_frames", 1), "runtime.uart_online.infer_step_frames")
    idle_sleep_ms = int(online_cfg.get("idle_sleep_ms", 1))
    log_every_n_frames = int(online_cfg.get("log_every_n_frames", 50))
    warmup_send = bool(online_cfg.get("send_zeros_before_ready", False))
    quiet = bool(online_cfg.get("quiet", False))
    debug_predict_trace = bool(online_cfg.get("debug_predict_trace", False))
    debug_uart_read_timing = bool(online_cfg.get("debug_uart_read_timing", False))
    debug_outer_rx = bool(online_cfg.get("debug_outer_rx", False))
    debug_outer_rx_only_abnormal = bool(online_cfg.get("debug_outer_rx_only_abnormal", False))
    debug_outer_rx_interval_warn_ms = float(online_cfg.get("debug_outer_rx_interval_warn_ms", 25.0))
    debug_tx_timing = bool(online_cfg.get("debug_tx_timing", False))
    debug_tx_only_abnormal = bool(online_cfg.get("debug_tx_only_abnormal", False))
    debug_tx_interval_min_warn_ms = float(online_cfg.get("debug_tx_interval_min_warn_ms", 180.0))
    debug_tx_interval_max_warn_ms = float(online_cfg.get("debug_tx_interval_max_warn_ms", 240.0))
    flush_rx_on_start = bool(online_cfg.get("flush_rx_on_start", True))
    startup_flush_empty_rounds = int(online_cfg.get("startup_flush_empty_rounds", 3))
    startup_flush_sleep_ms = int(online_cfg.get("startup_flush_sleep_ms", 10))
    feature_mode = get_feature_mode(cfg)

    input_value_type = str(online_cfg.get("input_value_type", uart_sender.value_type)).lower()
    input_byte_order = str(online_cfg.get("input_byte_order", uart_sender.byte_order)).lower()
    if uart_sender.outer_frame_enabled:
        parser = UartBundledValueFrameParser(
            outer_header=uart_sender.outer_header,
            outer_tail=uart_sender.outer_tail,
            inner_header=uart_sender.header,
            inner_tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
            outer_frame_count=uart_sender.outer_frame_count,
        )
    else:
        parser = UartValueFrameParser(
            header=uart_sender.header,
            tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
        )

    mean, scale = load_scaler_params(scaler_json_path)
    if len(mean) != window_size or len(scale) != window_size:
        raise RuntimeError("scaler length mismatch: need {}, got mean={}, scale={}".format(window_size, len(mean), len(scale)))

    nn, kpu, model_reloaded = ensure_kpu_cache(kmodel_path)

    # ring 的形状为 [通道数, 窗口长度]。
    # 每来一帧输入，就把 12 路值写到当前列位置，然后写指针前进。
    ring = empty_float((channel_count, window_size))
    write_idx = 0
    filled_frames = 0
    total_rx_frames = 0
    total_tx_frames = 0
    infer_round = 0

    # 这些临时数组反复复用，避免在 while True 中频繁分配内存。
    tmp_window = empty_float((window_size,))
    tmp_feature = empty_float((window_size,))
    tmp_scaled = empty_float((window_size,))
    sample3d = empty_float((1, 1, window_size))

    def online_print(*args):
        if not quiet:
            print(*args)

    online_print("uart_online_start: root={}".format(root))
    online_print(
        "uart_online_cfg: model_type=cnn, channels={}, window={}, infer_step_frames={}, input_type={}, input_order={}, model_reloaded={}".format(
            channel_count,
            window_size,
            infer_step_frames,
            input_value_type,
            input_byte_order,
            bool(model_reloaded),
        )
    )
    online_print("uart_online_feature_mode: {}".format(feature_mode))
    online_print(
        "uart_online_raw_anomaly: enabled={}, hit_count={}, clear_count={}".format(
            bool(raw_anomaly.enabled),
            int(raw_anomaly.hit_count),
            int(raw_anomaly.clear_count),
        )
    )
    online_print("uart_online_postprocessing: enabled={}, type={}".format(bool(postprocessor.enabled), postprocessor.kind))
    online_print("uart_online_full_gas_alarm:", full_gas_alarm.summary())
    if uart_sender.outer_frame_enabled:
        online_print(
            "uart_online_outer_frame_cfg: outer_frame_count={}, outer_header={}, outer_tail={}".format(
                uart_sender.outer_frame_count,
                " ".join("{:02X}".format(b) for b in uart_sender.outer_header),
                " ".join("{:02X}".format(b) for b in uart_sender.outer_tail),
            )
        )
    if flush_rx_on_start:
        flushed_bytes = drain_uart_rx(
            uart_sender.uart,
            empty_rounds=startup_flush_empty_rounds,
            sleep_between_ms=startup_flush_sleep_ms,
        )
        online_print(
            "uart_online_startup_flush: enabled=True, flushed_bytes={}, empty_rounds={}, sleep_ms={}".format(
                flushed_bytes,
                startup_flush_empty_rounds,
                startup_flush_sleep_ms,
            )
        )
    else:
        online_print("uart_online_startup_flush: enabled=False")

    session_start_us = now_us()
    first_rx_us = None
    last_infer_trigger_us = None
    last_uart_read_us = None
    last_outer_rx_us = None
    last_small_rx_us = None
    last_tx_us = None

    while True:
        raw = uart_sender.uart.read()
        rx_now_us = now_us()
        if debug_uart_read_timing and not quiet:
            read_interval_ms = -1.0
            if last_uart_read_us is not None:
                read_interval_ms = diff_us(rx_now_us, last_uart_read_us) / 1000.0
            raw_len = 0
            if raw:
                raw_len = len(raw)
            print(
                "uart_online_read: ts_ms={:.3f}, interval_ms={:.3f}, raw_bytes={}, has_data={}".format(
                    rx_now_us / 1000.0,
                    read_interval_ms,
                    raw_len,
                    bool(raw),
                )
            )
        last_uart_read_us = rx_now_us
        if raw:
            frames = parser.feed(raw)
            if not frames:
                continue
            if uart_sender.outer_frame_enabled:
                outer_count = int(uart_sender.outer_frame_count)
                parsed_outer_frames = len(frames) // outer_count
                if parsed_outer_frames > 0:
                    outer_interval_ms = -1.0
                    if last_outer_rx_us is not None:
                        outer_interval_ms = diff_us(rx_now_us, last_outer_rx_us) / 1000.0
                    if debug_outer_rx:
                        need_print_outer_rx = True
                        if debug_outer_rx_only_abnormal:
                            need_print_outer_rx = parsed_outer_frames > 1
                            if not need_print_outer_rx and outer_interval_ms >= 0.0:
                                need_print_outer_rx = outer_interval_ms >= debug_outer_rx_interval_warn_ms
                        if need_print_outer_rx:
                            online_print(
                                "uart_online_outer_rx: ts_ms={:.3f}, outer_frame_idx={}, batch_outer_frames={}, interval_ms={:.3f}, raw_bytes={}, parsed_small_frames={}".format(
                                    rx_now_us / 1000.0,
                                    (total_rx_frames + len(frames)) // outer_count,
                                    parsed_outer_frames,
                                    outer_interval_ms,
                                    len(raw),
                                    len(frames),
                                )
                            )
                        last_outer_rx_us = rx_now_us
            else:
                small_interval_ms = -1.0
                if last_small_rx_us is not None:
                    small_interval_ms = diff_us(rx_now_us, last_small_rx_us) / 1000.0
                if debug_outer_rx:
                    need_print_small_rx = True
                    if debug_outer_rx_only_abnormal:
                        need_print_small_rx = len(frames) > 1
                        if not need_print_small_rx and small_interval_ms >= 0.0:
                            need_print_small_rx = small_interval_ms >= debug_outer_rx_interval_warn_ms
                    if need_print_small_rx:
                        online_print(
                            "uart_online_small_rx: ts_ms={:.3f}, small_frame_idx={}, batch_small_frames={}, interval_ms={:.3f}, raw_bytes={}".format(
                                rx_now_us / 1000.0,
                                total_rx_frames + len(frames),
                                len(frames),
                                small_interval_ms,
                                len(raw),
                            )
                        )
                    last_small_rx_us = rx_now_us
            for values in frames:
                # 一帧输入对应 12 路同一时刻的采样值。
                total_rx_frames += 1
                if first_rx_us is None:
                    first_rx_us = now_us()
                for c in range(channel_count):
                    ring[c][write_idx] = float(values[c])
                write_idx += 1
                if write_idx >= window_size:
                    write_idx = 0
                just_became_ready = False
                if filled_frames < window_size:
                    # 尚未满窗时，还不能做有效推理。
                    filled_frames += 1
                    if filled_frames >= window_size:
                        # 第一次刚好满窗时，立刻触发首轮预测。
                        just_became_ready = True
                    else:
                        if warmup_send and uart_sender.enabled:
                            uart_sender.send_values_frame([0.0] * uart_sender.value_count)
                            total_tx_frames += 1
                        continue

                if not just_became_ready:
                    if ((total_rx_frames - window_size) % infer_step_frames) != 0:
                        # 满窗后不一定每帧都推理，可按 infer_step_frames 降低负载。
                        continue

                window_start = total_rx_frames - window_size + 1
                window_end = total_rx_frames
                if debug_predict_trace:
                    trigger_now_us = now_us()
                    elapsed_from_start_ms = diff_us(now_us(), session_start_us) / 1000.0
                    elapsed_from_first_rx_ms = -1.0
                    if first_rx_us is not None:
                        elapsed_from_first_rx_ms = diff_us(trigger_now_us, first_rx_us) / 1000.0
                    since_last_infer_ms = -1.0
                    if last_infer_trigger_us is not None:
                        since_last_infer_ms = diff_us(trigger_now_us, last_infer_trigger_us) / 1000.0
                    if uart_sender.outer_frame_enabled:
                        outer_count = int(uart_sender.outer_frame_count)
                        trigger_outer_frame = total_rx_frames // outer_count
                        window_outer_start = (window_start + outer_count - 1) // outer_count
                        window_outer_end = (window_end + outer_count - 1) // outer_count
                        online_print(
                            "uart_online_trigger: infer_round_next={}, rx_small_frame_idx={}, rx_outer_frame_idx={}, window_small=[{}, {}], window_outer=[{}, {}], first_ready={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
                                infer_round + 1,
                                total_rx_frames,
                                trigger_outer_frame,
                                window_start,
                                window_end,
                                window_outer_start,
                                window_outer_end,
                                just_became_ready,
                                elapsed_from_start_ms,
                                elapsed_from_first_rx_ms,
                                since_last_infer_ms,
                            )
                        )
                    else:
                        online_print(
                            "uart_online_trigger: infer_round_next={}, rx_small_frame_idx={}, window_small=[{}, {}], first_ready={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
                                infer_round + 1,
                                total_rx_frames,
                                window_start,
                                window_end,
                                just_became_ready,
                                elapsed_from_start_ms,
                                elapsed_from_first_rx_ms,
                                since_last_infer_ms,
                            )
                        )
                    last_infer_trigger_us = trigger_now_us

                preds = []
                raw_error_codes = []
                freq_total = 0.0
                t0 = now_us()
                for c in range(channel_count):
                    # 逐路展开窗口、标准化、推理，得到该通道的干度结果。
                    expand_ring_window(ring[c], write_idx, tmp_window)
                    raw_error_codes.append(raw_anomaly.update(c, tmp_window))
                    freq_total += mean_1d(tmp_window)
                    apply_feature_mode_1d(tmp_window, feature_mode, tmp_feature)
                    tmp_scaled[:] = (tmp_feature - mean) / scale
                    sample3d[0][0] = tmp_scaled
                    input_tensor = nn.from_numpy(sample3d)
                    kpu.set_input_tensor(0, input_tensor)
                    kpu.run()
                    output = kpu.get_output_tensor(0)
                    pred = float(output.to_numpy().reshape(-1)[0])
                    pred = postprocessor.update(c, pred)
                    preds.append(pred)
                    del output
                    del input_tensor

                # 输出协议仍固定为 12 路；若通道数不足则后面补 0。
                send_vals = []
                out_count = int(uart_sender.value_count)
                for i in range(out_count):
                    if i < len(preds):
                        send_vals.append(float(preds[i]))
                    else:
                        send_vals.append(0.0)
                update_full_gas_alarm_state(
                    full_gas_alarm,
                    preds,
                    freq_total / float(channel_count),
                    zero_guard_hit=False,
                )
                tx_now_us = now_us()
                tx_interval_ms = -1.0
                if last_tx_us is not None:
                    tx_interval_ms = diff_us(tx_now_us, last_tx_us) / 1000.0
                if raw_anomaly.enabled or full_gas_alarm.enabled:
                    packed_vals = []
                    for i in range(out_count):
                        code = RAW_ANOMALY_OK
                        if i < len(raw_error_codes):
                            code = raw_error_codes[i]
                        if code == RAW_ANOMALY_OK and full_gas_alarm.enabled and full_gas_alarm.alarm_on and i < len(preds):
                            code = FULL_GAS_ALARM_CODE
                        packed_vals.append(pack_alarm_dryness(code, send_vals[i], dryness_scale=uart_sender.scale))
                    uart_sender.send_raw_int_values_frame(packed_vals)
                else:
                    uart_sender.send_values_frame(send_vals)
                total_tx_frames += 1
                infer_round += 1
                if debug_tx_timing:
                    need_print_tx = True
                    if debug_tx_only_abnormal:
                        need_print_tx = False
                        if tx_interval_ms >= 0.0:
                            if tx_interval_ms < debug_tx_interval_min_warn_ms:
                                need_print_tx = True
                            elif tx_interval_ms > debug_tx_interval_max_warn_ms:
                                need_print_tx = True
                    if need_print_tx:
                        online_print(
                            "uart_online_tx: ts_ms={:.3f}, tx_small_frame_idx={}, infer_round={}, interval_since_last_tx_ms={:.3f}, first3={}".format(
                                tx_now_us / 1000.0,
                                total_tx_frames,
                                infer_round,
                                tx_interval_ms,
                                preds[:3],
                            )
                        )
                last_tx_us = tx_now_us
                infer_us = diff_us(now_us(), t0)
                if debug_predict_trace:
                    online_print(
                        "uart_online_result: infer_round={}, infer_ms={:.3f}, tx_small_frame_idx={}, first3={}, raw_error_codes={}, full_gas_alarm={}, alarm_reason={}".format(
                            infer_round,
                            infer_us / 1000.0,
                            total_tx_frames,
                            preds[:3],
                            raw_error_codes[:3],
                            bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                            full_gas_alarm.last_reason,
                        )
                    )

                if log_every_n_frames > 0 and (total_rx_frames % log_every_n_frames) == 0:
                    online_print(
                        "uart_online_stat: rx_frames={}, tx_frames={}, infer_round={}, infer_ms={:.3f}, first3={}, raw_error_codes={}, full_gas_alarm={}".format(
                            total_rx_frames,
                            total_tx_frames,
                            infer_round,
                            infer_us / 1000.0,
                            preds[:3],
                            raw_error_codes[:3],
                            bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                        )
                    )
                if infer_round % 20 == 0:
                    gc.collect()
        else:
            sleep_ms(idle_sleep_ms)


def run_online_uart_inference_cnn_lstm(cfg, root, uart_sender, kmodel_path, scaler_json_path):
    """
    在线串口 CNN-LSTM 推理模式。

    流程与离线 csv_cached 保持一致：
    1. 持续接收 12 路原始点并写入基础窗口环形缓冲。
    2. 每累计一个 base_step，就生成一个基础窗口特征。
    3. 再把最近 sequence_length 个基础窗口拼成序列样本。
    4. 序列满后按 sequence_step 触发一次推理。
    """
    runtime_cfg = cfg.get("runtime", {})
    online_cfg = get_runtime_section(runtime_cfg, "uart_online")
    alarm_cfg = get_runtime_section(runtime_cfg, "full_gas_alarm")
    data_cfg = cfg.get("data", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart_online mode cannot start.")

    window_size = require_positive_int(data_cfg.get("base_window_size", 500), "data.base_window_size")
    base_step = resolve_positive_step(data_cfg.get("base_step", None), window_size // 2, "data.base_step")
    seq_length = require_positive_int(data_cfg.get("sequence_length", 1), "data.sequence_length")
    seq_step = require_positive_int(data_cfg.get("sequence_step", 1), "data.sequence_step")
    if seq_length <= 1:
        raise RuntimeError("uart_online {} mode requires data.sequence_length > 1.".format(get_model_type(cfg)))

    channel_count = require_positive_int(online_cfg.get("channel_count", uart_sender.value_count), "runtime.uart_online.channel_count")
    infer_channel_count = require_positive_int(
        online_cfg.get("infer_channel_count", channel_count),
        "runtime.uart_online.infer_channel_count",
    )
    if infer_channel_count > channel_count:
        infer_channel_count = channel_count
    raw_anomaly = RawChannelAnomalyState(get_raw_anomaly_config(cfg), infer_channel_count)
    postprocessor = create_runtime_postprocessor(cfg, channel_count=infer_channel_count)
    full_gas_alarm = FullGasAlarmState(alarm_cfg)
    idle_sleep_ms = int(online_cfg.get("idle_sleep_ms", 1))
    log_every_n_frames = int(online_cfg.get("log_every_n_frames", 0))
    warmup_send = bool(online_cfg.get("send_zeros_before_ready", False))
    quiet = bool(online_cfg.get("quiet", False))
    debug_predict_trace = bool(online_cfg.get("debug_predict_trace", False))
    debug_uart_read_timing = bool(online_cfg.get("debug_uart_read_timing", False))
    debug_outer_rx = bool(online_cfg.get("debug_outer_rx", False))
    debug_outer_rx_only_abnormal = bool(online_cfg.get("debug_outer_rx_only_abnormal", False))
    debug_outer_rx_interval_warn_ms = float(online_cfg.get("debug_outer_rx_interval_warn_ms", 25.0))
    debug_tx_timing = bool(online_cfg.get("debug_tx_timing", False))
    debug_tx_only_abnormal = bool(online_cfg.get("debug_tx_only_abnormal", True))
    debug_tx_interval_min_warn_ms = float(online_cfg.get("debug_tx_interval_min_warn_ms", 180.0))
    debug_tx_interval_max_warn_ms = float(online_cfg.get("debug_tx_interval_max_warn_ms", 240.0))
    flush_rx_on_start = bool(online_cfg.get("flush_rx_on_start", True))
    startup_flush_empty_rounds = int(online_cfg.get("startup_flush_empty_rounds", 3))
    startup_flush_sleep_ms = int(online_cfg.get("startup_flush_sleep_ms", 10))
    feature_mode = get_feature_mode(cfg)
    model_type = get_model_type(cfg)
    uses_raw_aux = model_type == "cnn_tcn_seg3_soft_stats_moe"
    zero_guard_cfg = get_zero_guard_config(cfg)
    zero_guard_enabled = bool(zero_guard_cfg.get("enabled", False))
    zero_guard_output_value = float(zero_guard_cfg.get("output_value", 0.0))

    input_value_type = str(online_cfg.get("input_value_type", uart_sender.value_type)).lower()
    input_byte_order = str(online_cfg.get("input_byte_order", uart_sender.byte_order)).lower()
    if uart_sender.outer_frame_enabled:
        parser = UartBundledValueFrameParser(
            outer_header=uart_sender.outer_header,
            outer_tail=uart_sender.outer_tail,
            inner_header=uart_sender.header,
            inner_tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
            outer_frame_count=uart_sender.outer_frame_count,
        )
    else:
        parser = UartValueFrameParser(
            header=uart_sender.header,
            tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
        )

    mean, scale = load_scaler_params(scaler_json_path)
    if len(mean) != window_size or len(scale) != window_size:
        raise RuntimeError("scaler length mismatch: need {}, got mean={}, scale={}".format(window_size, len(mean), len(scale)))

    nn, kpu, model_reloaded = ensure_kpu_cache(kmodel_path)

    raw_ring = empty_float((infer_channel_count, window_size))
    raw_write_idx = 0
    raw_filled_frames = 0
    raw_frames_since_emit = 0

    seq_ring = empty_float((infer_channel_count, seq_length, window_size))
    raw_seq_ring = empty_float((infer_channel_count, seq_length, window_size)) if uses_raw_aux else None
    zero_seq_ring = empty_float((infer_channel_count, seq_length, window_size)) if zero_guard_enabled else None
    channel_raw_error_codes = [RAW_ANOMALY_OK] * infer_channel_count
    seq_write_idx = 0
    seq_filled = 0
    seq_windows_since_infer = 0

    total_rx_frames = 0
    total_tx_frames = 0
    base_window_count = 0
    infer_round = 0

    tmp_window = empty_float((window_size,))
    tmp_feature = empty_float((window_size,))
    tmp_scaled = empty_float((window_size,))
    tmp_seq = empty_float((seq_length, window_size))
    tmp_raw_seq = empty_float((seq_length, window_size)) if uses_raw_aux else None
    tmp_zero_seq = empty_float((seq_length, window_size)) if zero_guard_enabled else None
    zero_guard_states = [ZeroGuardState(zero_guard_cfg) for _ in range(infer_channel_count)] if zero_guard_enabled else None
    sample3d = empty_float((1, seq_length, window_size))
    raw_sample3d = empty_float((1, seq_length, window_size)) if uses_raw_aux else None

    def online_print(*args):
        if not quiet:
            print(*args)

    def send_zero_frame():
        nonlocal total_tx_frames
        uart_sender.send_values_frame([0.0] * int(uart_sender.value_count))
        total_tx_frames += 1

    online_print("uart_online_start: root={}".format(root))
    online_print(
        "uart_online_cfg: model_type={}, input_channels={}, infer_channels={}, output_values={}, window={}, base_step={}, sequence_length={}, sequence_step={}, input_type={}, input_order={}, model_reloaded={}".format(
            get_model_type(cfg),
            channel_count,
            infer_channel_count,
            int(uart_sender.value_count),
            window_size,
            base_step,
            seq_length,
            seq_step,
            input_value_type,
            input_byte_order,
            bool(model_reloaded),
        )
    )
    online_print("uart_online_feature_mode: {}".format(feature_mode))
    online_print(
        "uart_online_raw_anomaly: enabled={}, hit_count={}, clear_count={}".format(
            bool(raw_anomaly.enabled),
            int(raw_anomaly.hit_count),
            int(raw_anomaly.clear_count),
        )
    )
    online_print("uart_online_postprocessing: enabled={}, type={}".format(bool(postprocessor.enabled), postprocessor.kind))
    online_print("uart_online_full_gas_alarm:", full_gas_alarm.summary())
    online_print(
        "uart_online_zero_guard: enabled={}, output_value={}, enter_freq={}, exit_freq={}, enter_windows={}, exit_windows={}".format(
            bool(zero_guard_enabled),
            zero_guard_output_value,
            float(zero_guard_cfg.get("freq_enter_threshold", 480000.0)),
            float(zero_guard_cfg.get("freq_exit_threshold", 500000.0)),
            int(zero_guard_cfg.get("enter_consecutive_windows", 3)),
            int(zero_guard_cfg.get("exit_consecutive_windows", 3)),
        )
    )
    if uart_sender.outer_frame_enabled:
        online_print(
            "uart_online_outer_frame_cfg: outer_frame_count={}, outer_header={}, outer_tail={}".format(
                uart_sender.outer_frame_count,
                " ".join("{:02X}".format(b) for b in uart_sender.outer_header),
                " ".join("{:02X}".format(b) for b in uart_sender.outer_tail),
            )
        )
    if flush_rx_on_start:
        flushed_bytes = drain_uart_rx(
            uart_sender.uart,
            empty_rounds=startup_flush_empty_rounds,
            sleep_between_ms=startup_flush_sleep_ms,
        )
        online_print(
            "uart_online_startup_flush: enabled=True, flushed_bytes={}, empty_rounds={}, sleep_ms={}".format(
                flushed_bytes,
                startup_flush_empty_rounds,
                startup_flush_sleep_ms,
            )
        )
    else:
        online_print("uart_online_startup_flush: enabled=False")

    session_start_us = now_us()
    first_rx_us = None
    last_infer_trigger_us = None
    last_uart_read_us = None
    last_outer_rx_us = None
    last_small_rx_us = None
    last_tx_us = None

    while True:
        raw = uart_sender.uart.read()
        rx_now_us = now_us()
        if debug_uart_read_timing and not quiet:
            read_interval_ms = -1.0
            if last_uart_read_us is not None:
                read_interval_ms = diff_us(rx_now_us, last_uart_read_us) / 1000.0
            raw_len = 0
            if raw:
                raw_len = len(raw)
            print(
                "uart_online_read: ts_ms={:.3f}, interval_ms={:.3f}, raw_bytes={}, has_data={}".format(
                    rx_now_us / 1000.0,
                    read_interval_ms,
                    raw_len,
                    bool(raw),
                )
            )
        last_uart_read_us = rx_now_us
        if not raw:
            sleep_ms(idle_sleep_ms)
            continue

        frames = parser.feed(raw)
        if not frames:
            continue

        if uart_sender.outer_frame_enabled:
            outer_count = int(uart_sender.outer_frame_count)
            parsed_outer_frames = len(frames) // outer_count
            if parsed_outer_frames > 0:
                outer_interval_ms = -1.0
                if last_outer_rx_us is not None:
                    outer_interval_ms = diff_us(rx_now_us, last_outer_rx_us) / 1000.0
                if debug_outer_rx:
                    need_print_outer_rx = True
                    if debug_outer_rx_only_abnormal:
                        need_print_outer_rx = parsed_outer_frames > 1
                        if not need_print_outer_rx and outer_interval_ms >= 0.0:
                            need_print_outer_rx = outer_interval_ms >= debug_outer_rx_interval_warn_ms
                    if need_print_outer_rx:
                        online_print(
                            "uart_online_outer_rx: ts_ms={:.3f}, outer_frame_idx={}, batch_outer_frames={}, interval_ms={:.3f}, raw_bytes={}, parsed_small_frames={}".format(
                                rx_now_us / 1000.0,
                                (total_rx_frames + len(frames)) // outer_count,
                                parsed_outer_frames,
                                outer_interval_ms,
                                len(raw),
                                len(frames),
                            )
                        )
                last_outer_rx_us = rx_now_us
        else:
            small_interval_ms = -1.0
            if last_small_rx_us is not None:
                small_interval_ms = diff_us(rx_now_us, last_small_rx_us) / 1000.0
            if debug_outer_rx:
                need_print_small_rx = True
                if debug_outer_rx_only_abnormal:
                    need_print_small_rx = len(frames) > 1
                    if not need_print_small_rx and small_interval_ms >= 0.0:
                        need_print_small_rx = small_interval_ms >= debug_outer_rx_interval_warn_ms
                if need_print_small_rx:
                    online_print(
                        "uart_online_small_rx: ts_ms={:.3f}, small_frame_idx={}, batch_small_frames={}, interval_ms={:.3f}, raw_bytes={}".format(
                            rx_now_us / 1000.0,
                            total_rx_frames + len(frames),
                            len(frames),
                            small_interval_ms,
                            len(raw),
                        )
                    )
            last_small_rx_us = rx_now_us

        for values in frames:
            total_rx_frames += 1
            if first_rx_us is None:
                first_rx_us = now_us()

            for c in range(infer_channel_count):
                raw_ring[c][raw_write_idx] = float(values[c])
            raw_write_idx += 1
            if raw_write_idx >= window_size:
                raw_write_idx = 0

            emit_base_window = False
            first_base_window = False
            if raw_filled_frames < window_size:
                raw_filled_frames += 1
                if raw_filled_frames >= window_size:
                    emit_base_window = True
                    first_base_window = True
                    raw_frames_since_emit = 0
                else:
                    if warmup_send and uart_sender.enabled:
                        send_zero_frame()
                    continue
            else:
                raw_frames_since_emit += 1
                if raw_frames_since_emit >= base_step:
                    emit_base_window = True
                    raw_frames_since_emit = 0
                else:
                    if warmup_send and uart_sender.enabled and seq_filled < seq_length:
                        send_zero_frame()
                    continue

            if not emit_base_window:
                continue

            freq_total = 0.0
            for c in range(infer_channel_count):
                expand_ring_window(raw_ring[c], raw_write_idx, tmp_window)
                channel_raw_error_codes[c] = raw_anomaly.update(c, tmp_window)
                freq_total += mean_1d(tmp_window)
                if zero_guard_enabled:
                    # 中文注释：0 干度保护必须使用未去均值的原始窗口，和 PC 端 zero_guard 保持一致。
                    zero_seq_ring[c][seq_write_idx] = tmp_window
                apply_feature_mode_1d(tmp_window, feature_mode, tmp_feature)
                tmp_scaled[:] = (tmp_feature - mean) / scale
                seq_ring[c][seq_write_idx] = tmp_scaled
                if uses_raw_aux:
                    raw_seq_ring[c][seq_write_idx] = tmp_feature

            seq_write_idx += 1
            if seq_write_idx >= seq_length:
                seq_write_idx = 0
            base_window_count += 1

            first_seq_ready = False
            if seq_filled < seq_length:
                seq_filled += 1
                if seq_filled >= seq_length:
                    first_seq_ready = True
                    seq_windows_since_infer = 0
                else:
                    if warmup_send and uart_sender.enabled:
                        send_zero_frame()
                    continue
            else:
                seq_windows_since_infer += 1
                if seq_windows_since_infer < seq_step:
                    if warmup_send and uart_sender.enabled:
                        send_zero_frame()
                    continue
                seq_windows_since_infer = 0

            raw_window_end = total_rx_frames
            raw_window_start = raw_window_end - window_size + 1
            seq_raw_start = raw_window_end - (window_size - 1) - (seq_length - 1) * base_step
            if debug_predict_trace:
                trigger_now_us = now_us()
                elapsed_from_start_ms = diff_us(trigger_now_us, session_start_us) / 1000.0
                elapsed_from_first_rx_ms = -1.0
                if first_rx_us is not None:
                    elapsed_from_first_rx_ms = diff_us(trigger_now_us, first_rx_us) / 1000.0
                since_last_infer_ms = -1.0
                if last_infer_trigger_us is not None:
                    since_last_infer_ms = diff_us(trigger_now_us, last_infer_trigger_us) / 1000.0
                online_print(
                    "uart_online_trigger: infer_round_next={}, rx_small_frame_idx={}, base_window_idx={}, sequence_ready={}, first_base_window={}, first_sequence_ready={}, raw_window=[{}, {}], sequence_raw_start={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
                        infer_round + 1,
                        total_rx_frames,
                        base_window_count,
                        seq_filled >= seq_length,
                        first_base_window,
                        first_seq_ready,
                        raw_window_start,
                        raw_window_end,
                        seq_raw_start,
                        elapsed_from_start_ms,
                        elapsed_from_first_rx_ms,
                        since_last_infer_ms,
                    )
                )
                last_infer_trigger_us = trigger_now_us

            preds = []
            zero_guard_hits = 0
            t0 = now_us()
            for c in range(infer_channel_count):
                expand_sequence_ring(seq_ring[c], seq_write_idx, tmp_seq)
                if zero_guard_enabled:
                    expand_sequence_ring(zero_seq_ring[c], seq_write_idx, tmp_zero_seq)
                    guard_hit, guard_votes, guard_features = is_zero_guard_hit(
                        tmp_zero_seq,
                        tmp_seq,
                        zero_guard_cfg,
                        state=zero_guard_states[c],
                    )
                    if guard_hit:
                        pred = postprocessor.update(c, zero_guard_output_value, zero_guard_hit=True)
                        preds.append(pred)
                        zero_guard_hits += 1
                        if debug_predict_trace and c < 3:
                            online_print(
                                "uart_online_zero_guard_hit: infer_round_next={}, channel={}, votes={}, features={}".format(
                                    infer_round + 1,
                                    c,
                                    int(guard_votes),
                                    {
                                        "freq_mean": round(float(guard_features.get("freq_mean", 0.0)), 3),
                                        "diff_p95_abs": round(float(guard_features.get("diff_p95_abs", 0.0)), 3),
                                        "win_range_mean": round(float(guard_features.get("win_range_mean", 0.0)), 3),
                                        "win_std_mean": round(float(guard_features.get("win_std_mean", 0.0)), 3),
                                        "absz_mean": round(float(guard_features.get("absz_mean", 0.0)), 6),
                                        "zero_identity": bool(guard_features.get("zero_identity", False)),
                                        "zero_confidence_high": bool(guard_features.get("zero_confidence_high", False)),
                                        "zero_guard_state": bool(guard_features.get("zero_guard_state", False)),
                                        "enter_count": int(guard_features.get("enter_count", 0)),
                                        "exit_count": int(guard_features.get("exit_count", 0)),
                                    },
                                )
                            )
                        continue
                sample3d[0] = tmp_seq
                input_tensor = nn.from_numpy(sample3d)
                kpu.set_input_tensor(0, input_tensor)
                raw_input_tensor = None
                if uses_raw_aux:
                    expand_sequence_ring(raw_seq_ring[c], seq_write_idx, tmp_raw_seq)
                    raw_sample3d[0] = tmp_raw_seq
                    raw_input_tensor = nn.from_numpy(raw_sample3d)
                    kpu.set_input_tensor(1, raw_input_tensor)
                kpu.run()
                output = kpu.get_output_tensor(0)
                pred = float(output.to_numpy().reshape(-1)[0])
                pred = postprocessor.update(c, pred)
                preds.append(pred)
                del output
                del input_tensor
                if raw_input_tensor is not None:
                    del raw_input_tensor

            send_vals = []
            out_count = int(uart_sender.value_count)
            for i in range(out_count):
                if i < len(preds):
                    send_vals.append(float(preds[i]))
                else:
                    send_vals.append(0.0)
            update_full_gas_alarm_state(
                full_gas_alarm,
                preds,
                freq_total / float(infer_channel_count),
                zero_guard_hit=zero_guard_hits > 0,
            )
            tx_now_us = now_us()
            tx_interval_ms = -1.0
            if last_tx_us is not None:
                tx_interval_ms = diff_us(tx_now_us, last_tx_us) / 1000.0
            if raw_anomaly.enabled or full_gas_alarm.enabled:
                packed_vals = []
                for i in range(out_count):
                    code = RAW_ANOMALY_OK
                    if i < len(channel_raw_error_codes):
                        code = channel_raw_error_codes[i]
                    if code == RAW_ANOMALY_OK and full_gas_alarm.enabled and full_gas_alarm.alarm_on and i < len(preds):
                        code = FULL_GAS_ALARM_CODE
                    packed_vals.append(pack_alarm_dryness(code, send_vals[i], dryness_scale=uart_sender.scale))
                uart_sender.send_raw_int_values_frame(packed_vals)
            else:
                uart_sender.send_values_frame(send_vals)
            total_tx_frames += 1
            infer_round += 1
            if debug_tx_timing:
                need_print_tx = True
                if debug_tx_only_abnormal:
                    need_print_tx = False
                    if tx_interval_ms >= 0.0:
                        if tx_interval_ms < debug_tx_interval_min_warn_ms:
                            need_print_tx = True
                        elif tx_interval_ms > debug_tx_interval_max_warn_ms:
                            need_print_tx = True
                if need_print_tx:
                    online_print(
                        "uart_online_tx: ts_ms={:.3f}, tx_small_frame_idx={}, infer_round={}, interval_since_last_tx_ms={:.3f}, first3={}".format(
                            tx_now_us / 1000.0,
                            total_tx_frames,
                            infer_round,
                            tx_interval_ms,
                            preds[:3],
                        )
                    )
            last_tx_us = tx_now_us
            infer_us = diff_us(now_us(), t0)
            if debug_predict_trace:
                online_print(
                    "uart_online_result: infer_round={}, infer_ms={:.3f}, tx_small_frame_idx={}, zero_guard_hits={}, first3={}, raw_error_codes={}, full_gas_alarm={}, alarm_reason={}".format(
                        infer_round,
                        infer_us / 1000.0,
                        total_tx_frames,
                        zero_guard_hits,
                        preds[:3],
                        channel_raw_error_codes[:3],
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                        full_gas_alarm.last_reason,
                    )
                )

            if log_every_n_frames > 0 and (total_rx_frames % log_every_n_frames) == 0:
                online_print(
                    "uart_online_stat: rx_frames={}, tx_frames={}, base_window_count={}, infer_round={}, infer_ms={:.3f}, zero_guard_hits={}, first3={}, raw_error_codes={}, full_gas_alarm={}".format(
                        total_rx_frames,
                        total_tx_frames,
                        base_window_count,
                        infer_round,
                        infer_us / 1000.0,
                        zero_guard_hits,
                        preds[:3],
                        channel_raw_error_codes[:3],
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                    )
                )
            if infer_round % 20 == 0:
                gc.collect()


def run_online_uart_inference(cfg, root, uart_sender, kmodel_path, scaler_json_path):
    # 按配置中的模型类型自动分流在线推理逻辑。
    model_type = get_model_type(cfg)
    if model_type == "cnn":
        return run_online_uart_inference_cnn(cfg, root, uart_sender, kmodel_path, scaler_json_path)
    if model_type in {"cnn_lstm", "cnn_tcn", "cnn_tcn_seg3_soft_stats_moe"}:
        return run_online_uart_inference_cnn_lstm(cfg, root, uart_sender, kmodel_path, scaler_json_path)
    raise RuntimeError("Unsupported model.type for uart_online: {}".format(model_type))


def run_uart_echo(root, cfg, uart_sender):
    """
    串口环路测试模式。

    用于最基础的链路联调：
    不走协议解析，不走模型推理，收到什么原样回什么。
    """
    runtime_cfg = cfg.get("runtime", {})
    echo_cfg = runtime_cfg.get("uart_echo", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart echo mode cannot start.")

    # 空闲时的轮询等待时间。
    # 这里不是阻塞式中断接收，而是不断 read() 轮询，
    # 所以给一个很小的 sleep，避免空转占满 CPU。
    idle_sleep_ms = int(echo_cfg.get("idle_sleep_ms", 1))
    # 每收到多少包打印一次统计信息，便于观察是否持续在收发。
    log_every_n_packets = int(echo_cfg.get("log_every_n_packets", 50))
    # 调试开关：打开后会把每一包按十六进制打印出来。
    # 串口流量大时不建议长期打开，否则打印本身会拖慢速度。
    print_hex = bool(echo_cfg.get("print_hex", False))

    rx_packets = 0
    rx_bytes = 0
    tx_bytes = 0

    print("uart_echo_start: root={}".format(root))
    print(
        "uart_echo_cfg: baudrate={}, idle_sleep_ms={}, log_every_n_packets={}, print_hex={}".format(
            uart_sender.uart.baudrate() if hasattr(uart_sender.uart, "baudrate") else "unknown",
            idle_sleep_ms,
            log_every_n_packets,
            print_hex,
        )
    )

    while True:
        # 从 UART 接收缓冲区取出当前已经到达的数据。
        # read() 返回的是这一时刻能读到的原始字节，不做任何协议解析。
        data = uart_sender.uart.read()
        if not data:
            sleep_ms(idle_sleep_ms)
            continue

        rx_packets += 1
        rx_bytes += len(data)

        try:
            # 环路测试的核心逻辑：
            # 收到什么字节，就把同样的字节原样写回去。
            # 不改帧头帧尾，不改长度，也不做预测计算。
            written = uart_sender.uart.write(data)
        except Exception as exc:
            print("WARN: uart echo write failed:", exc)
            continue

        if written is None:
            written = 0
        tx_bytes += int(written)

        if print_hex:
            # 打开详细日志时，直接打印本包的十六进制内容，
            # 可用于和串口助手/单片机抓到的数据逐字节对比。
            hex_text = " ".join("{:02X}".format(b) for b in data)
            print("uart_echo_packet: bytes={} hex={}".format(len(data), hex_text))
        elif log_every_n_packets > 0 and (rx_packets % log_every_n_packets) == 0:
            print(
                "uart_echo_stat: packets={}, rx_bytes={}, tx_bytes={}".format(
                    rx_packets, rx_bytes, tx_bytes
                )
            )


def run_uart_return_every_n_frames(root, cfg, uart_sender):
    """
    按帧计数的回发测试模式。

    设计目标：
    1. 单片机持续按固定协议发帧给 K230。
    2. K230 只做收帧和计数，不做预测。
    3. 每累计 N 帧，仅把第 N 帧原样回发一次。
    """

    runtime_cfg = cfg.get("runtime", {})
    return_cfg = runtime_cfg.get("uart_frame_return", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart frame return mode cannot start.")

    return_every_n = require_positive_int(
        return_cfg.get("return_every_n_frames", 500),
        "runtime.uart_frame_return.return_every_n_frames",
    )
    idle_sleep_ms = int(return_cfg.get("idle_sleep_ms", 1))
    log_every_n_frames = int(return_cfg.get("log_every_n_frames", 100))
    print_hex = bool(return_cfg.get("print_hex", False))
    strict_protocol = bool(return_cfg.get("strict_protocol", True))
    fixed_frame_len = int(return_cfg.get("fixed_frame_len", 52))
    return_inner_frame_when_outer_enabled = bool(return_cfg.get("return_inner_frame_when_outer_enabled", True))
    return_inner_frame_index = int(return_cfg.get("return_inner_frame_index", -1))

    if strict_protocol:
        if uart_sender.outer_frame_enabled:
            parser = UartBundledRawFrameParser(
                outer_header=uart_sender.outer_header,
                outer_tail=uart_sender.outer_tail,
                inner_header=uart_sender.header,
                inner_tail=uart_sender.tail,
                value_count=uart_sender.value_count,
                outer_frame_count=uart_sender.outer_frame_count,
            )
        else:
            parser = UartRawFrameParser(
                header=uart_sender.header,
                tail=uart_sender.tail,
                value_count=uart_sender.value_count,
            )
    else:
        parser = UartFixedLengthParser(frame_len=fixed_frame_len)

    total_rx_frames = 0
    total_tx_frames = 0
    total_rx_bytes = 0
    total_tx_bytes = 0

    print("uart_frame_return_start: root={}".format(root))
    print(
        "uart_frame_return_cfg: return_every_n_frames={}, idle_sleep_ms={}, log_every_n_frames={}, print_hex={}, strict_protocol={}, fixed_frame_len={}".format(
            return_every_n,
            idle_sleep_ms,
            log_every_n_frames,
            print_hex,
            strict_protocol,
            fixed_frame_len,
        )
    )
    if uart_sender.outer_frame_enabled:
        print(
            "uart_frame_return_outer_frame_cfg: outer_frame_count={}, outer_frame_len={}, outer_header={}, outer_tail={}, return_inner_frame_when_outer_enabled={}, return_inner_frame_index={}".format(
                uart_sender.outer_frame_count,
                uart_sender.outer_frame_len,
                " ".join("{:02X}".format(b) for b in uart_sender.outer_header),
                " ".join("{:02X}".format(b) for b in uart_sender.outer_tail),
                return_inner_frame_when_outer_enabled,
                return_inner_frame_index,
            )
        )

    while True:
        data = uart_sender.uart.read()
        if not data:
            sleep_ms(idle_sleep_ms)
            continue

        total_rx_bytes += len(data)
        frames = parser.feed(data)
        if not frames:
            continue

        for frame in frames:
            total_rx_frames += 1
            if print_hex:
                hex_text = " ".join("{:02X}".format(b) for b in frame)
                print("uart_frame_rx: idx={} bytes={} hex={}".format(total_rx_frames, len(frame), hex_text))

            if (total_rx_frames % return_every_n) == 0:
                tx_frame = frame
                if uart_sender.outer_frame_enabled and return_inner_frame_when_outer_enabled:
                    inner_count = int(uart_sender.outer_frame_count)
                    inner_len = int(uart_sender.inner_frame_len)
                    idx = int(return_inner_frame_index)
                    if idx < 0:
                        idx = inner_count + idx
                    if idx < 0:
                        idx = 0
                    if idx >= inner_count:
                        idx = inner_count - 1
                    outer_header_len = len(uart_sender.outer_header)
                    start = outer_header_len + idx * inner_len
                    end = start + inner_len
                    tx_frame = frame[start:end]
                try:
                    written = uart_sender.uart.write(tx_frame)
                except Exception as exc:
                    print("WARN: uart frame return write failed:", exc)
                    continue

                if written is None:
                    written = 0
                total_tx_frames += 1
                total_tx_bytes += int(written)

                if print_hex:
                    print("uart_frame_tx: idx={} bytes={}".format(total_rx_frames, written))
                else:
                    print(
                        "uart_frame_return_hit: rx_frame_idx={}, tx_frames={}, rx_bytes={}, tx_bytes={}".format(
                            total_rx_frames,
                            total_tx_frames,
                            total_rx_bytes,
                            total_tx_bytes,
                        )
                    )
            elif log_every_n_frames > 0 and (total_rx_frames % log_every_n_frames) == 0:
                print(
                    "uart_frame_return_stat: rx_frames={}, tx_frames={}, rx_bytes={}, tx_bytes={}".format(
                        total_rx_frames,
                        total_tx_frames,
                        total_rx_bytes,
                        total_tx_bytes,
                    )
                )


def run_uart_debug_ack(root, cfg, uart_sender):
    """
    调试 ACK 模式。

    设计目标：
    1. 每收到 1 个完整大帧，就立刻回 1 个调试小帧。
    2. 返回内容全部使用原始整数，便于单片机直接核对计数与时间戳。
    3. 不跑模型，不依赖窗口和推理触发逻辑。
    """
    runtime_cfg = cfg.get("runtime", {})
    ack_cfg = runtime_cfg.get("uart_debug_ack", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart debug ack mode cannot start.")

    idle_sleep_ms = int(ack_cfg.get("idle_sleep_ms", 1))
    log_every_n_frames = int(ack_cfg.get("log_every_n_frames", 20))
    print_hex = bool(ack_cfg.get("print_hex", False))
    strict_protocol = bool(ack_cfg.get("strict_protocol", True))
    fixed_frame_len = int(ack_cfg.get("fixed_frame_len", 524))
    flush_rx_on_start = bool(ack_cfg.get("flush_rx_on_start", True))
    startup_flush_empty_rounds = int(ack_cfg.get("startup_flush_empty_rounds", 3))
    startup_flush_sleep_ms = int(ack_cfg.get("startup_flush_sleep_ms", 10))
    ack_magic = int(ack_cfg.get("ack_magic", 9001))

    if strict_protocol:
        if uart_sender.outer_frame_enabled:
            parser = UartBundledRawFrameParser(
                outer_header=uart_sender.outer_header,
                outer_tail=uart_sender.outer_tail,
                inner_header=uart_sender.header,
                inner_tail=uart_sender.tail,
                value_count=uart_sender.value_count,
                outer_frame_count=uart_sender.outer_frame_count,
            )
        else:
            parser = UartRawFrameParser(
                header=uart_sender.header,
                tail=uart_sender.tail,
                value_count=uart_sender.value_count,
            )
    else:
        parser = UartFixedLengthParser(frame_len=fixed_frame_len)

    total_rx_frames = 0
    total_tx_frames = 0
    total_rx_bytes = 0
    ack_seq = 0

    print("uart_debug_ack_start: root={}".format(root))
    print(
        "uart_debug_ack_cfg: idle_sleep_ms={}, log_every_n_frames={}, print_hex={}, strict_protocol={}, fixed_frame_len={}, ack_magic={}".format(
            idle_sleep_ms,
            log_every_n_frames,
            print_hex,
            strict_protocol,
            fixed_frame_len,
            ack_magic,
        )
    )
    if uart_sender.outer_frame_enabled:
        print(
            "uart_debug_ack_outer_frame_cfg: outer_frame_count={}, outer_frame_len={}, outer_header={}, outer_tail={}".format(
                uart_sender.outer_frame_count,
                uart_sender.outer_frame_len,
                " ".join("{:02X}".format(b) for b in uart_sender.outer_header),
                " ".join("{:02X}".format(b) for b in uart_sender.outer_tail),
            )
        )
    if flush_rx_on_start:
        flushed_bytes = drain_uart_rx(
            uart_sender.uart,
            empty_rounds=startup_flush_empty_rounds,
            sleep_between_ms=startup_flush_sleep_ms,
        )
        print(
            "uart_debug_ack_startup_flush: enabled=True, flushed_bytes={}, empty_rounds={}, sleep_ms={}".format(
                flushed_bytes,
                startup_flush_empty_rounds,
                startup_flush_sleep_ms,
            )
        )
    else:
        print("uart_debug_ack_startup_flush: enabled=False")

    while True:
        data = uart_sender.uart.read()
        if not data:
            sleep_ms(idle_sleep_ms)
            continue

        total_rx_bytes += len(data)
        frames = parser.feed(data)
        if not frames:
            continue

        for frame in frames:
            total_rx_frames += 1
            ack_seq += 1
            rx_outer_frame_idx = int(total_rx_frames)
            if uart_sender.outer_frame_enabled:
                rx_small_frame_idx = int(total_rx_frames * uart_sender.outer_frame_count)
            else:
                rx_small_frame_idx = int(total_rx_frames)
            board_ticks_ms = clamp_int32(now_us() // 1000)
            ack_values = [
                ack_magic,
                ack_seq,
                board_ticks_ms,
                rx_outer_frame_idx,
                rx_small_frame_idx,
                clamp_int32(total_rx_bytes),
                len(frame),
                1,
                0,
                0,
                0,
                0,
            ]
            uart_sender.send_raw_int_values_frame(ack_values)
            total_tx_frames += 1

            if print_hex:
                hex_text = " ".join("{:02X}".format(b) for b in frame)
                print("uart_debug_ack_rx: idx={} bytes={} hex={}".format(total_rx_frames, len(frame), hex_text))

            if log_every_n_frames > 0 and (total_rx_frames % log_every_n_frames) == 0:
                print(
                    "uart_debug_ack_stat: rx_outer_frames={}, rx_small_frames={}, tx_ack_frames={}, rx_bytes={}, ack_seq={}".format(
                        total_rx_frames,
                        rx_small_frame_idx,
                        total_tx_frames,
                        total_rx_bytes,
                        ack_seq,
                    )
                )


def write_predictions(path, y_true, y_pred):
    # 离线模式下可选地把预测结果写回 CSV，便于后续人工核对。
    ensure_dir(dirname(path))
    with open(path, "w") as f:
        f.write("sample_id,true_label,prediction\n")
        for i in range(len(y_pred)):
            f.write("{},{},{}\n".format(i, float(y_true[i]), float(y_pred[i])))

def path_with_kmodel_name(output_path, kmodel_path):
    # 板端输出文件统一带上 kmodel 文件名，避免多次测试结果混在一起分不清。
    out_text = norm_path(output_path)
    kmodel_name = norm_path(kmodel_path).split("/")[-1]
    if "." in kmodel_name:
        kmodel_name = kmodel_name.rsplit(".", 1)[0]

    out_dir = dirname(out_text)
    out_name = out_text.split("/")[-1]
    if "." in out_name:
        stem, ext = out_name.rsplit(".", 1)
        final_name = stem + "__" + kmodel_name + "." + ext
    else:
        final_name = out_name + "__" + kmodel_name
    if out_dir:
        return join_path(out_dir, final_name)
    return final_name


def run_kmodel_inference(kmodel_path, X_scaled, uart_sender=None):
    # 旧版全量推理函数，当前主要保留作兼容与对照。
    import nncase_runtime as nn  # type: ignore

    kpu = nn.kpu()
    kpu.load_kmodel(kmodel_path)

    preds = []
    infer_us_total = 0
    for i in range(X_scaled.shape[0]):
        sample = astype_float_array(X_scaled[i])
        sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
        input_tensor = nn.from_numpy(sample)
        kpu.set_input_tensor(0, input_tensor)
        t0 = now_us()
        kpu.run()
        t1 = now_us()
        infer_us_total += diff_us(t1, t0)
        output = kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        preds.append(pred)
        if uart_sender is not None:
            uart_sender.send_scaled_prediction(pred)
        del output
        del input_tensor
        if (i + 1) % 64 == 0:
            gc.collect()
    return as_float_array(preds), infer_us_total


def safe_metric_mae(y_true, y_pred):
    # 手写 MAE，避免依赖额外统计库。
    total = 0.0
    count = 0
    for i in range(len(y_pred)):
        t = float(y_true[i])
        p = float(y_pred[i])
        if t == t:
            d = p - t
            if d < 0:
                d = -d
            total += d
            count += 1
    if count == 0:
        return float("nan")
    return total / float(count)


def safe_metric_rmse(y_true, y_pred):
    # 手写 RMSE，便于板端直接计算评估指标。
    total = 0.0
    count = 0
    for i in range(len(y_pred)):
        t = float(y_true[i])
        p = float(y_pred[i])
        if t == t:
            d = p - t
            total += d * d
            count += 1
    if count == 0:
        return float("nan")
    return float(np.sqrt(total / float(count)))


def detect_root():
    # 中文注释：依次尝试多个候选目录，找到实际的应用根目录。
    # 旧版本曾以 `k230_config.json` 作为根目录锚点，但现在配置已经迁移到 `configs/` 下，
    # 如果继续找旧文件，板端很容易把 cwd 误判成 root，随后读取 `configs/*.json` 时触发 ENOENT。
    candidates = []
    try:
        candidates.append(norm_path(os.getcwd()))
    except Exception:
        pass
    here = globals().get("__file__", "")
    if here:
        here = norm_path(here)
        if "/" in here:
            candidates.append(dirname(here))
    candidates.append("/sdcard/raw_cnn_k230")
    candidates.append("/sdcard")

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        if (
            exists(join_path(c, "run_k230_infer.py"))
            and exists(join_path(c, "configs"))
        ):
            return c
        if exists(join_path(c, "configs/auto_start_config.json")):
            return c
        if exists(join_path(c, "configs/k230_config_cnn_tcn.json")):
            return c
    return ordered[0] if ordered else "."


def main():
    # 中文注释：新版唯一入口交给 runtime.app。
    # 本文件里的旧函数仍保留给兼容后端调用，避免一次性重写板端推理细节。
    import runtime.app as runtime_app
    runtime_app.main(config_path=OVERRIDE_CONFIG_PATH)
    return

    # 统一入口：先读配置，再根据 runtime.mode 分流到不同模式。
    # 平时切模式优先改 k230_config.json，不要直接改这里的分支。
    # 整个脚本的统一入口：
    # 先读取配置，再根据 runtime.mode 决定进入哪一种运行模式。
    root = detect_root()
    cli_args = []
    if sys is not None:
        try:
            cli_args = list(sys.argv[1:])
        except Exception:
            cli_args = []
    config_path = resolve_runtime_config_path(root, cli_args)
    cfg = load_json(config_path)
    paths = cfg["paths"]
    runtime_cfg = cfg.get("runtime", {})
    csv_cfg = get_runtime_section(runtime_cfg, "csv_cached")
    mode = normalize_runtime_mode(runtime_cfg.get("mode", "csv_cached"))
    uart_cfg = dict(cfg.get("uart", {}))
    if mode == "uart_online":
        online_cfg = get_runtime_section(runtime_cfg, "uart_online")
        uart_cfg["quiet"] = bool(online_cfg.get("quiet", False))
    uart_sender = UartDrynessSender(uart_cfg)
    max_samples = csv_cfg.get("max_samples", runtime_cfg.get("max_samples", None))
    if max_samples is not None:
        max_samples = require_positive_int(max_samples, "runtime.csv_cached.max_samples")
    infer_batch_size = csv_cfg.get("infer_batch_size", runtime_cfg.get("infer_batch_size", uart_cfg.get("value_count", 12)))
    infer_batch_size = require_positive_int(infer_batch_size, "runtime.csv_cached.infer_batch_size")
    write_csv = bool(csv_cfg.get("write_predictions_csv", runtime_cfg.get("write_predictions_csv", False)))

    kmodel_path = join_path(root, paths["kmodel"])
    scaler_json_path = join_path(root, paths["scaler_json"])
    pred_csv = join_path(root, paths["predictions_csv"])
    pred_csv = path_with_kmodel_name(pred_csv, kmodel_path)

    # 启动时先打印当前模型信息，避免上板测试时分不清正在跑哪一版 kmodel。
    print("=== K230 Runtime Model ===")
    print("config_name:", cfg.get("name", ""))
    print("config_path:", config_path)
    print("mode:", mode)
    print("kmodel:", kmodel_path)
    print("scaler_json:", scaler_json_path)
    print("predictions_csv:", pred_csv)

    # 运行模式说明：
    # 1. uart_online: 串口实时接收 12 路数据，满窗后做在线推理。
    # 2. uart_echo:   串口环路测试模式，收到什么就原样发回什么。
    # 3. uart_debug_ack: 每收到 1 个大帧就回 1 个调试 ACK 小帧。
    # 4. csv_cached:  用本地 CSV 做离线推理调试。
    if mode == "uart_online":
        run_online_uart_inference(
            cfg=cfg,
            root=root,
            uart_sender=uart_sender,
            kmodel_path=kmodel_path,
            scaler_json_path=scaler_json_path,
        )
        return
    if mode == "uart_frame_return":
        run_uart_return_every_n_frames(root=root, cfg=cfg, uart_sender=uart_sender)
        return
    if mode == "uart_echo":
        # 当前大将军测试串口通断和环路时，走这个分支。
        # 这个模式完全不依赖模型、CSV、标准化参数。
        run_uart_echo(root=root, cfg=cfg, uart_sender=uart_sender)
        return
    if mode == "uart_debug_ack":
        run_uart_debug_ack(root=root, cfg=cfg, uart_sender=uart_sender)
        return
    if mode != "csv_cached":
        raise ValueError("Unsupported runtime.mode: " + str(mode))

    t_start = now_us()
    X_scaled, y_all, rebuilt = ensure_dataset_cache(cfg, root, max_samples, scaler_json_path)
    X_raw_aux = None
    if get_model_type(cfg) == "cnn_tcn_seg3_soft_stats_moe":
        X_raw_aux = RUNTIME_CACHE.get("X_raw_aux", None)
    if rebuilt:
        print("dataset_cache_rebuilt_samples:", int(X_scaled.shape[0]))
    else:
        print("dataset_cache_hit_samples:", int(X_scaled.shape[0]))

    start_idx, count = acquire_infer_range(int(X_scaled.shape[0]), infer_batch_size)
    y_batch = collect_labels_range(y_all, start_idx, count)
    postprocessor = create_runtime_postprocessor(cfg, channel_count=1)
    y_pred, infer_us, model_reloaded = run_kmodel_inference_cached(
        kmodel_path, X_scaled, start_idx, count, uart_sender=uart_sender, X_raw_aux=X_raw_aux, postprocessor=postprocessor
    )
    if uart_sender.enabled:
        uart_sender.flush_pending()
    t_end = now_us()

    if write_csv:
        write_predictions(pred_csv, y_batch, y_pred)
    total_us = diff_us(t_end, t_start)
    mae = safe_metric_mae(y_batch, y_pred)
    rmse = safe_metric_rmse(y_batch, y_pred)

    print("=== K230 Raw+CNN Inference ===")
    print("root:", root)
    print("config_path:", config_path)
    print("mode:", mode)
    print("kmodel:", kmodel_path)
    print("dataset_total_samples:", int(X_scaled.shape[0]))
    print("infer_batch_size:", int(count))
    print("infer_start_idx:", int(start_idx))
    print("model_reloaded:", bool(model_reloaded))
    print("postprocessing:", postprocessor.kind if postprocessor.enabled else "none")
    print("input_shape:", tuple(X_scaled.shape[1:]))
    print("model_infer_time_sec:", infer_us / 1_000_000.0)
    print("model_infer_time_per_sample_ms:", infer_us / 1000.0 / float(count))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("write_predictions_csv:", bool(write_csv))
    if write_csv:
        print("prediction_csv:", pred_csv)
    print("first_10_predictions:", y_pred[:10].tolist())
    if uart_sender.enabled:
        print("uart_sent_frames:", int(uart_sender.send_count))
        print("uart_send_errors:", int(uart_sender.error_count))


if __name__ == "__main__":
    main()
