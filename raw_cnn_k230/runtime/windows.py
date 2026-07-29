"""物理通道窗口层。"""

from runtime import features
from runtime import numeric


class ChannelWindowBank:
    """中文注释：基础窗口维护接口。"""

    def __init__(self, channel_count):
        self.channel_count = int(channel_count)

    def update(self, frame):
        return []


class OnlineWindowBank:
    """中文注释：多输入在线模式的原始窗口和特征窗口维护器。"""

    def __init__(
        self,
        input_contexts,
        window_size,
        base_step,
        feature_mode,
        max_seq_length,
        zero_guard_enabled=False,
    ):
        self.input_contexts = input_contexts
        self.window_size = int(window_size)
        self.base_step = int(base_step)
        self.feature_mode = feature_mode
        self.max_seq_length = int(max_seq_length)
        self.zero_guard_enabled = bool(zero_guard_enabled)

        self.raw_ring = numeric.empty_float((len(input_contexts), self.window_size))
        self.raw_write_idx = 0
        self.raw_filled_frames = 0
        self.raw_frames_since_emit = 0
        self.base_window_count = 0

        self.tmp_window = numeric.empty_float((self.window_size,))
        self.tmp_feature_map = {}
        for input_ctx in input_contexts:
            self.tmp_feature_map[input_ctx["name"]] = numeric.empty_float((self.window_size,))

        if self.zero_guard_enabled:
            self.zero_mean_seq_ring = numeric.empty_float((len(input_contexts), self.max_seq_length))
        else:
            self.zero_mean_seq_ring = None
        self.zero_seq_write_idx = 0
        self.zero_seq_filled = 0

        self.last_first_base_window = False
        self.last_freq_mean = 0.0
        self.last_freq_mean_by_input = {}
        self.last_diff_std_by_input = {}

    def push_values(self, values, status_ctx):
        """中文注释：写入一帧物理输入；返回本帧是否触发了新的基础窗。"""
        for input_idx, input_ctx in enumerate(self.input_contexts):
            self.raw_ring[input_idx][self.raw_write_idx] = float(values[int(input_ctx["source_index"])])

        self.raw_write_idx += 1
        if self.raw_write_idx >= self.window_size:
            self.raw_write_idx = 0

        self.last_first_base_window = False
        if self.raw_filled_frames < self.window_size:
            self.raw_filled_frames += 1
            if self.raw_filled_frames < self.window_size:
                return False
            self.last_first_base_window = True
            self.raw_frames_since_emit = 0
        else:
            self.raw_frames_since_emit += 1
            if self.raw_frames_since_emit < self.base_step:
                return False
            self.raw_frames_since_emit = 0

        self._emit_base_window(status_ctx)
        return True

    def _emit_base_window(self, status_ctx):
        """中文注释：从环形缓存展开基础窗，并生成特征窗口和状态输入。"""
        freq_total = 0.0
        for input_idx, input_ctx in enumerate(self.input_contexts):
            numeric.expand_ring_window(self.raw_ring[input_idx], self.raw_write_idx, self.tmp_window)
            source_index = int(input_ctx["source_index"])
            status_ctx.update_raw_anomaly(source_index, self.tmp_window)
            channel_mean = numeric.mean_1d(self.tmp_window)
            channel_diff_std = numeric.diff_std_1d(self.tmp_window)
            self.last_freq_mean_by_input[input_ctx["name"]] = channel_mean
            self.last_diff_std_by_input[input_ctx["name"]] = channel_diff_std
            freq_total += channel_mean
            if self.zero_guard_enabled:
                self.zero_mean_seq_ring[input_idx][self.zero_seq_write_idx] = channel_mean
            features.apply_feature_mode_1d(
                self.tmp_window,
                self.feature_mode,
                self.tmp_feature_map[input_ctx["name"]],
            )

        self.last_freq_mean = freq_total / float(len(self.input_contexts))
        if self.zero_guard_enabled:
            self.zero_seq_write_idx += 1
            if self.zero_seq_write_idx >= self.max_seq_length:
                self.zero_seq_write_idx = 0
            if self.zero_seq_filled < self.max_seq_length:
                self.zero_seq_filled += 1
        self.base_window_count += 1

    def get_feature_window(self, input_name):
        """中文注释：读取指定输入路当前基础特征窗。"""
        return self.tmp_feature_map[input_name]

    def get_zero_guard_freq_mean(self, input_idx):
        """中文注释：返回最近序列窗口的原始频率均值。"""
        values = self.zero_mean_seq_ring[int(input_idx)]
        count = int(self.zero_seq_filled)
        total = 0.0
        for i in range(count):
            total += float(values[i])
        if count <= 0:
            return 0.0
        return total / float(count)

    def get_last_freq_mean(self, input_name):
        return float(self.last_freq_mean_by_input.get(input_name, self.last_freq_mean))

    def get_last_diff_std(self, input_name):
        return float(self.last_diff_std_by_input.get(input_name, 0.0))
