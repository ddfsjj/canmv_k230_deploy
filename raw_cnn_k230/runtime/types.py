"""统一运行框架的数据类型。

这些类型用于固定输入层、推理层、输出层之间的接口形态。
当前实现先复用旧推理后端，后续逐步把旧逻辑搬入 runtime 时，
这些对象就是稳定边界。
"""


class ChannelFrame:
    """中文注释：一次输入帧，包含同一时刻的多路物理通道值。"""

    def __init__(self, values, timestamp_ms=None):
        self.values = values
        self.timestamp_ms = timestamp_ms


class ChannelWindow:
    """中文注释：某个物理通道当前可用于推理的窗口。"""

    def __init__(self, channel, raw_window=None, processed_window=None, ready=False):
        self.channel = int(channel)
        self.raw_window = raw_window
        self.processed_window = processed_window
        self.ready = bool(ready)


class NamedPrediction:
    """中文注释：推理层输出的命名预测值，不直接绑定串口协议。"""

    def __init__(self, name, value, source_channel=0, output_slot=0, ready=True):
        self.name = str(name)
        self.value = float(value)
        self.source_channel = int(source_channel)
        self.output_slot = int(output_slot)
        self.ready = bool(ready)


class ChannelStatus:
    """中文注释：物理通道状态，例如原始异常码和零点保护命中状态。"""

    def __init__(self, channel, raw_anomaly_code=0, zero_guard_hit=False):
        self.channel = int(channel)
        self.raw_anomaly_code = int(raw_anomaly_code)
        self.zero_guard_hit = bool(zero_guard_hit)
