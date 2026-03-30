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
            if sleep_v > 0:
                sleep_ms(sleep_v)
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
    return "raw"


def get_feature_mode(cfg):
    preprocessing_cfg = cfg.get("preprocessing", {})
    return normalize_feature_mode(preprocessing_cfg.get("feature_mode", "raw"))


def apply_feature_mode_1d(src_window, feature_mode, out_window):
    mode = normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        mean_value = float(np.sum(src_window) / float(len(src_window)))
        out_window[:] = src_window - mean_value
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
    cached_X = RUNTIME_CACHE.get("X_scaled", None)
    cached_y = RUNTIME_CACHE.get("y", None)
    if (
        RUNTIME_CACHE.get("dataset_key", None) == cache_key
        and cached_X is not None
        and cached_y is not None
        and int(cached_X.shape[0]) > 0
    ):
        return cached_X, cached_y, False

    print("dataset_cache_miss_rebuild")
    X, y = build_dataset(cfg, root, max_samples=max_samples)
    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found in test_data.")
    X_scaled = scale_features(X, scaler_json_path)
    del X
    gc.collect()

    RUNTIME_CACHE["dataset_key"] = cache_key
    RUNTIME_CACHE["X_scaled"] = X_scaled
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


def run_kmodel_inference_cached(kmodel_path, X_scaled, start_idx, count, uart_sender=None):
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
        idx += 1
        if idx >= total:
            idx = 0
        if (i + 1) % 64 == 0:
            gc.collect()
    return as_float_array(preds), infer_us_total, model_reloaded


def run_online_uart_inference(cfg, root, uart_sender, kmodel_path, scaler_json_path):
    """
    在线串口推理模式。

    目标流程：
    1. 从单片机接收一帧 12 路输入。
    2. 写入 12 路环形缓冲。
    3. 当每路都积累满一个窗口后，逐路推理。
    4. 把 12 路预测结果再按同样协议打包发回去。
    """
    runtime_cfg = cfg.get("runtime", {})
    online_cfg = runtime_cfg.get("online_uart", {})
    data_cfg = cfg.get("data", {})

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; online uart mode cannot start.")

    window_size = require_positive_int(data_cfg.get("base_window_size", 500), "data.base_window_size")
    seq_length = require_positive_int(data_cfg.get("sequence_length", 1), "data.sequence_length")
    if seq_length != 1:
        raise RuntimeError("online uart mode currently requires data.sequence_length = 1.")

    channel_count = require_positive_int(online_cfg.get("channel_count", uart_sender.value_count), "runtime.online_uart.channel_count")
    infer_step_frames = require_positive_int(online_cfg.get("infer_step_frames", 1), "runtime.online_uart.infer_step_frames")
    idle_sleep_ms = int(online_cfg.get("idle_sleep_ms", 1))
    log_every_n_frames = int(online_cfg.get("log_every_n_frames", 50))
    warmup_send = bool(online_cfg.get("send_zeros_before_ready", False))
    debug_predict_trace = bool(online_cfg.get("debug_predict_trace", False))
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

    print("online_uart_start: root={}".format(root))
    print(
        "online_uart_cfg: channels={}, window={}, infer_step_frames={}, input_type={}, input_order={}, model_reloaded={}".format(
            channel_count,
            window_size,
            infer_step_frames,
            input_value_type,
            input_byte_order,
            bool(model_reloaded),
        )
    )
    print("online_uart_feature_mode: {}".format(feature_mode))
    if uart_sender.outer_frame_enabled:
        print(
            "online_uart_outer_frame_cfg: outer_frame_count={}, outer_header={}, outer_tail={}".format(
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
        print(
            "online_uart_startup_flush: enabled=True, flushed_bytes={}, empty_rounds={}, sleep_ms={}".format(
                flushed_bytes,
                startup_flush_empty_rounds,
                startup_flush_sleep_ms,
            )
        )
    else:
        print("online_uart_startup_flush: enabled=False")

    session_start_us = now_us()
    first_rx_us = None
    last_infer_trigger_us = None

    while True:
        raw = uart_sender.uart.read()
        if raw:
            frames = parser.feed(raw)
            if not frames:
                continue
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
                        print(
                            "online_uart_trigger: infer_round_next={}, rx_small_frame_idx={}, rx_outer_frame_idx={}, window_small=[{}, {}], window_outer=[{}, {}], first_ready={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
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
                        print(
                            "online_uart_trigger: infer_round_next={}, rx_small_frame_idx={}, window_small=[{}, {}], first_ready={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
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
                t0 = now_us()
                for c in range(channel_count):
                    # 逐路展开窗口、标准化、推理，得到该通道的干度结果。
                    expand_ring_window(ring[c], write_idx, tmp_window)
                    apply_feature_mode_1d(tmp_window, feature_mode, tmp_feature)
                    tmp_scaled[:] = (tmp_feature - mean) / scale
                    sample3d[0][0] = tmp_scaled
                    input_tensor = nn.from_numpy(sample3d)
                    kpu.set_input_tensor(0, input_tensor)
                    kpu.run()
                    output = kpu.get_output_tensor(0)
                    pred = float(output.to_numpy().reshape(-1)[0])
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
                uart_sender.send_values_frame(send_vals)
                total_tx_frames += 1
                infer_round += 1
                infer_us = diff_us(now_us(), t0)
                if debug_predict_trace:
                    print(
                        "online_uart_result: infer_round={}, infer_ms={:.3f}, tx_small_frame_idx={}, first3={}".format(
                            infer_round,
                            infer_us / 1000.0,
                            total_tx_frames,
                            preds[:3],
                        )
                    )

                if log_every_n_frames > 0 and (total_rx_frames % log_every_n_frames) == 0:
                    print(
                        "online_uart_stat: rx_frames={}, tx_frames={}, infer_round={}, infer_ms={:.3f}, first3={}".format(
                            total_rx_frames,
                            total_tx_frames,
                            infer_round,
                            infer_us / 1000.0,
                            preds[:3],
                        )
                    )
                if infer_round % 20 == 0:
                    gc.collect()
        else:
            sleep_ms(idle_sleep_ms)


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
    # 依次尝试多个候选目录，找到实际的应用根目录。
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
        if exists(join_path(c, "k230_config.json")):
            return c
    return ordered[0] if ordered else "."


def main():
    # 整个脚本的统一入口：
    # 先读取配置，再根据 runtime.mode 决定进入哪一种运行模式。
    root = detect_root()
    cfg = load_json(join_path(root, "k230_config.json"))
    paths = cfg["paths"]
    runtime_cfg = cfg.get("runtime", {})
    uart_cfg = cfg.get("uart", {})
    uart_sender = UartDrynessSender(uart_cfg)
    mode = str(runtime_cfg.get("mode", "csv_cached")).lower()
    max_samples = runtime_cfg.get("max_samples", None)
    if max_samples is not None:
        max_samples = require_positive_int(max_samples, "runtime.max_samples")
    infer_batch_size = runtime_cfg.get("infer_batch_size", uart_cfg.get("value_count", 12))
    infer_batch_size = require_positive_int(infer_batch_size, "runtime.infer_batch_size")
    write_csv = bool(runtime_cfg.get("write_predictions_csv", False))

    kmodel_path = join_path(root, paths["kmodel"])
    scaler_json_path = join_path(root, paths["scaler_json"])
    pred_csv = join_path(root, paths["predictions_csv"])

    # 运行模式说明：
    # 1. uart_online: 串口实时接收 12 路数据，满窗后做在线推理。
    # 2. uart_echo:   串口环路测试模式，收到什么就原样发回什么。
    # 3. uart_debug_ack: 每收到 1 个大帧就回 1 个调试 ACK 小帧。
    # 4. csv_cached:  用本地 CSV 做离线推理调试。
    if mode in {"uart_online", "online_uart"}:
        run_online_uart_inference(
            cfg=cfg,
            root=root,
            uart_sender=uart_sender,
            kmodel_path=kmodel_path,
            scaler_json_path=scaler_json_path,
        )
        return
    if mode in {"uart_frame_return", "frame_return"}:
        run_uart_return_every_n_frames(root=root, cfg=cfg, uart_sender=uart_sender)
        return
    if mode in {"uart_echo", "echo"}:
        # 当前大将军测试串口通断和环路时，走这个分支。
        # 这个模式完全不依赖模型、CSV、标准化参数。
        run_uart_echo(root=root, cfg=cfg, uart_sender=uart_sender)
        return
    if mode in {"uart_debug_ack", "debug_ack", "ack"}:
        run_uart_debug_ack(root=root, cfg=cfg, uart_sender=uart_sender)
        return

    t_start = now_us()
    X_scaled, y_all, rebuilt = ensure_dataset_cache(cfg, root, max_samples, scaler_json_path)
    if rebuilt:
        print("dataset_cache_rebuilt_samples:", int(X_scaled.shape[0]))
    else:
        print("dataset_cache_hit_samples:", int(X_scaled.shape[0]))

    start_idx, count = acquire_infer_range(int(X_scaled.shape[0]), infer_batch_size)
    y_batch = collect_labels_range(y_all, start_idx, count)
    y_pred, infer_us, model_reloaded = run_kmodel_inference_cached(
        kmodel_path, X_scaled, start_idx, count, uart_sender=uart_sender
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
    print("mode:", mode)
    print("kmodel:", kmodel_path)
    print("dataset_total_samples:", int(X_scaled.shape[0]))
    print("infer_batch_size:", int(count))
    print("infer_start_idx:", int(start_idx))
    print("model_reloaded:", bool(model_reloaded))
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
