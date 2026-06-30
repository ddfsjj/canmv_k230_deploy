"""输入源层。"""

from runtime import uart as runtime_uart


class InputSource:
    """中文注释：通用输入源接口。"""

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class UartReadBatch:
    """中文注释：一次 UART 读取后解析出的输入小帧批次。"""

    def __init__(self, frames, raw_len, rx_now_us):
        self.frames = frames
        self.raw_len = int(raw_len)
        self.rx_now_us = int(rx_now_us)


def build_uart_value_parser(base_module, uart_sender, channel_count, online_cfg):
    """中文注释：按当前配置创建 UART 输入解析器。"""
    input_value_type = str(online_cfg.get("input_value_type", "int32")).lower()
    input_byte_order = str(online_cfg.get("input_byte_order", uart_sender.byte_order)).lower()
    if uart_sender.outer_frame_enabled:
        return runtime_uart.UartBundledValueFrameParser(
            outer_header=uart_sender.outer_header,
            outer_tail=uart_sender.outer_tail,
            inner_header=uart_sender.header,
            inner_tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
            outer_frame_count=uart_sender.outer_frame_count,
        )
    return runtime_uart.UartValueFrameParser(
        header=uart_sender.header,
        tail=uart_sender.tail,
        value_count=channel_count,
        value_type=input_value_type,
        byte_order=input_byte_order,
    )


class UartOnlineInputReader:
    """中文注释：封装 UART read、帧解析和接收时序日志。"""

    def __init__(self, base_module, uart_sender, channel_count, online_cfg, online_print, quiet=False):
        self.base = base_module
        self.uart_sender = uart_sender
        self.channel_count = int(channel_count)
        self.online_print = online_print
        self.quiet = bool(quiet)
        self.parser = build_uart_value_parser(base_module, uart_sender, channel_count, online_cfg)

        self.debug_uart_read_timing = bool(online_cfg.get("debug_uart_read_timing", False))
        self.debug_outer_rx = bool(online_cfg.get("debug_outer_rx", False))
        self.debug_outer_rx_only_abnormal = bool(online_cfg.get("debug_outer_rx_only_abnormal", False))
        self.debug_outer_rx_interval_warn_ms = float(online_cfg.get("debug_outer_rx_interval_warn_ms", 25.0))

        self.last_uart_read_us = None
        self.last_outer_rx_us = None
        self.last_small_rx_us = None

    def read_batch(self, total_rx_frames):
        """中文注释：读取 UART 并解析出一批小帧；无数据或未成帧时返回 None。"""
        raw = self.uart_sender.uart.read()
        rx_now_us = self.base.now_us()
        self._log_uart_read(rx_now_us, raw)
        self.last_uart_read_us = rx_now_us

        if not raw:
            return None

        frames = self.parser.feed(raw)
        if not frames:
            return UartReadBatch(frames=[], raw_len=len(raw), rx_now_us=rx_now_us)

        self._log_parsed_frames(rx_now_us, raw, frames, total_rx_frames)
        return UartReadBatch(frames=frames, raw_len=len(raw), rx_now_us=rx_now_us)

    def _log_uart_read(self, rx_now_us, raw):
        if not self.debug_uart_read_timing or self.quiet:
            return
        read_interval_ms = -1.0
        if self.last_uart_read_us is not None:
            read_interval_ms = self.base.diff_us(rx_now_us, self.last_uart_read_us) / 1000.0
        raw_len = len(raw) if raw else 0
        self.online_print(
            "uart_online_read: ts_ms={:.3f}, interval_ms={:.3f}, raw_bytes={}, has_data={}".format(
                rx_now_us / 1000.0,
                read_interval_ms,
                raw_len,
                bool(raw),
            )
        )

    def _log_parsed_frames(self, rx_now_us, raw, frames, total_rx_frames):
        if self.uart_sender.outer_frame_enabled:
            self._log_outer_frames(rx_now_us, raw, frames, total_rx_frames)
        else:
            self._log_small_frames(rx_now_us, raw, frames, total_rx_frames)

    def _log_outer_frames(self, rx_now_us, raw, frames, total_rx_frames):
        outer_count = int(self.uart_sender.outer_frame_count)
        parsed_outer_frames = len(frames) // outer_count
        if parsed_outer_frames <= 0:
            return
        outer_interval_ms = -1.0
        if self.last_outer_rx_us is not None:
            outer_interval_ms = self.base.diff_us(rx_now_us, self.last_outer_rx_us) / 1000.0
        if self.debug_outer_rx:
            need_print_outer_rx = True
            if self.debug_outer_rx_only_abnormal:
                need_print_outer_rx = parsed_outer_frames > 1
                if not need_print_outer_rx and outer_interval_ms >= 0.0:
                    need_print_outer_rx = outer_interval_ms >= self.debug_outer_rx_interval_warn_ms
            if need_print_outer_rx:
                self.online_print(
                    "uart_online_outer_rx: ts_ms={:.3f}, outer_frame_idx={}, batch_outer_frames={}, interval_ms={:.3f}, raw_bytes={}, parsed_small_frames={}".format(
                        rx_now_us / 1000.0,
                        (int(total_rx_frames) + len(frames)) // outer_count,
                        parsed_outer_frames,
                        outer_interval_ms,
                        len(raw),
                        len(frames),
                    )
                )
        self.last_outer_rx_us = rx_now_us

    def _log_small_frames(self, rx_now_us, raw, frames, total_rx_frames):
        small_interval_ms = -1.0
        if self.last_small_rx_us is not None:
            small_interval_ms = self.base.diff_us(rx_now_us, self.last_small_rx_us) / 1000.0
        if self.debug_outer_rx:
            need_print_small_rx = True
            if self.debug_outer_rx_only_abnormal:
                need_print_small_rx = len(frames) > 1
                if not need_print_small_rx and small_interval_ms >= 0.0:
                    need_print_small_rx = small_interval_ms >= self.debug_outer_rx_interval_warn_ms
            if need_print_small_rx:
                self.online_print(
                    "uart_online_small_rx: ts_ms={:.3f}, small_frame_idx={}, batch_small_frames={}, interval_ms={:.3f}, raw_bytes={}".format(
                        rx_now_us / 1000.0,
                        int(total_rx_frames) + len(frames),
                        len(frames),
                        small_interval_ms,
                        len(raw),
                    )
                )
        self.last_small_rx_us = rx_now_us
