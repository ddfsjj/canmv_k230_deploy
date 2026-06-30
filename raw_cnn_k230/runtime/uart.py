"""UART 和帧协议适配层。"""

try:
    import ustruct as struct  # type: ignore
except ImportError:
    import struct  # type: ignore

try:
    from machine import UART, FPIOA  # type: ignore
except ImportError:
    UART = None  # type: ignore
    FPIOA = None  # type: ignore

from runtime import protocol


def parse_frame_bytes(raw, default_bytes):
    """中文注释：兼容列表、十六进制字符串和单整数三种帧字节配置。"""
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


class UartDrynessSender:
    """中文注释：统一封装 UART 初始化、12 路返回帧编码和发送。"""

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
        self.header = parse_frame_bytes(uart_cfg.get("header", [0x55, 0xAA]), [0x55, 0xAA])
        self.tail = parse_frame_bytes(uart_cfg.get("tail", [0xFC, 0xCF]), [0xFC, 0xCF])
        self.outer_frame_enabled = bool(uart_cfg.get("outer_frame_enabled", False))
        self.outer_frame_count = int(uart_cfg.get("outer_frame_count", 10))
        self.outer_header = parse_frame_bytes(uart_cfg.get("outer_header", [0xF7, 0x7F]), [0xF7, 0x7F])
        self.outer_tail = parse_frame_bytes(uart_cfg.get("outer_tail", [0xFA, 0xAF]), [0xFA, 0xAF])
        if self.byte_order not in {"little", "big"}:
            self.byte_order = "little"
        if self.value_type not in {"int32", "float32"}:
            self.value_type = "int32"
        if self.outer_frame_count <= 0:
            self.outer_frame_count = 10
        self.inner_frame_len = len(self.header) + self.value_count * 4 + len(self.tail)
        self.outer_payload_len = self.outer_frame_count * self.inner_frame_len
        self.outer_frame_len = len(self.outer_header) + self.outer_payload_len + len(self.outer_tail)

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

            bits_const = UART.SEVENBITS if bits == 7 else UART.EIGHTBITS
            parity_key = str(parity).lower()
            if parity_key == "even":
                parity_const = UART.PARITY_EVEN
            elif parity_key == "odd":
                parity_const = UART.PARITY_ODD
            else:
                parity_const = UART.PARITY_NONE
            stop_const = UART.STOPBITS_TWO if stop == 2 else UART.STOPBITS_ONE

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

    def _encode_frame(self, values, apply_scale=True):
        """中文注释：把一组数值编码为 header + 12*4 payload + tail。"""
        payload = bytearray()
        int_fmt = ">i" if self.byte_order == "big" else "<i"
        float_fmt = ">f" if self.byte_order == "big" else "<f"
        for i in range(self.value_count):
            fval = 0.0
            if i < len(values):
                raw = values[i]
                if protocol.is_finite_number(raw):
                    fval = float(raw)
            if self.value_type == "float32":
                payload.extend(struct.pack(float_fmt, float(fval)))
            else:
                if apply_scale:
                    packed_value = protocol.clamp_int32(int(round(float(fval) * self.scale)))
                else:
                    packed_value = protocol.clamp_int32(int(round(float(fval))))
                payload.extend(struct.pack(int_fmt, int(packed_value)))
        frame = bytearray(self.header)
        frame.extend(payload)
        frame.extend(self.tail)
        return frame

    def send_scaled_prediction(self, pred_value):
        """中文注释：离线逐样本预测时先缓存，攒够 value_count 后发送一帧。"""
        if not self.enabled or self.uart is None:
            return
        v = 0.0
        if protocol.is_finite_number(pred_value):
            v = float(pred_value)
        self.pending_values.append(v)
        if len(self.pending_values) < self.value_count:
            return
        values = self.pending_values[: self.value_count]
        del self.pending_values[: self.value_count]
        self._send_values(values)

    def _send_values(self, values):
        frame = self._encode_frame(values, apply_scale=True)
        try:
            self.uart.write(frame)
            self.send_count += 1
        except Exception as exc:
            self.error_count += 1
            if not self.quiet:
                print("WARN: UART send failed:", exc)

    def send_raw_int_values_frame(self, values):
        """中文注释：直接发送原始整数，不再乘 predict_scale。"""
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
        """中文注释：发送一整组预测值。"""
        if not self.enabled or self.uart is None:
            return
        self._send_values(values)

    def flush_pending(self):
        """中文注释：把缓存里不足一帧的预测值补零发出。"""
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


class UartValueFrameParser:
    """中文注释：从小帧字节流中解析固定数量的 int32/float32 通道值。"""

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
            payload = self._buf[header_len:tail_start]
            try:
                out.append(self._decode_payload(payload))
            except Exception:
                pass
            self._buf = bytearray(self._buf[self._frame_len :])
        return out


class UartBundledValueFrameParser:
    """中文注释：解析外层大帧，再拆成多个小帧通道值。"""

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


class UartRawFrameParser:
    """中文注释：只提取完整小帧原始字节，不解析 payload。"""

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
            out.append(bytes(self._buf[: self._frame_len]))
            self._buf = bytearray(self._buf[self._frame_len :])
        return out


class UartBundledRawFrameParser:
    """中文注释：解析外层大帧并返回完整大帧原始字节。"""

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


class UartFixedLengthParser:
    """中文注释：固定长度分包器，用于弱校验联调。"""

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


def build_sender(uart_cfg):
    """中文注释：统一创建 runtime 发送器。"""
    return UartDrynessSender(uart_cfg)


def make_frame_bytes(uart_cfg, values):
    """中文注释：给测试和调试使用，把 12 路值编码成当前返回帧字节。"""
    sender = UartDrynessSender(dict(uart_cfg, enabled=False))
    return sender._encode_frame(values, apply_scale=True)
