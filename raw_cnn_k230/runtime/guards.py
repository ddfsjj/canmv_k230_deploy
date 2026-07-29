"""运行期状态保护逻辑。

这个模块承载 raw anomaly、zero guard、预测后处理和满气报警状态机。
它只包含纯状态逻辑，不依赖 UART、KPU 或旧入口脚本。
"""

try:
    import ulab.numpy as np  # type: ignore
except ImportError:
    import numpy as np  # type: ignore

from runtime import protocol


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


def median_list(values):
    count = int(len(values))
    if count <= 0:
        return 0.0
    ordered = sorted([float(v) for v in values])
    mid = count // 2
    if (count % 2) == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def get_raw_anomaly_config(cfg):
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
    if not bool(alarm_cfg.get("enabled", False)):
        return protocol.RAW_ANOMALY_OK

    count = int(len(raw_window))
    if count <= 0:
        return protocol.RAW_ANOMALY_OK

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
        return protocol.RAW_ANOMALY_ALL_ZERO

    raw_min = _cfg_float(alarm_cfg, "raw_min", None)
    if raw_min is not None and bool(alarm_cfg.get("raw_range_enabled", True)) and min_v < raw_min:
        return protocol.RAW_ANOMALY_LOW

    raw_max = _cfg_float(alarm_cfg, "raw_max", None)
    if raw_max is not None and bool(alarm_cfg.get("raw_range_enabled", True)) and max_v > raw_max:
        return protocol.RAW_ANOMALY_HIGH

    if all_same and first != 0.0 and bool(alarm_cfg.get("stuck_enabled", True)):
        return protocol.RAW_ANOMALY_STUCK

    spike_max_diff = _cfg_float(alarm_cfg, "spike_max_diff", None)
    if spike_max_diff is not None and bool(alarm_cfg.get("spike_enabled", True)) and max_diff > spike_max_diff:
        return protocol.RAW_ANOMALY_SPIKE

    return protocol.RAW_ANOMALY_OK


class RawChannelAnomalyState:
    """每个物理输入通道独立防抖，异常码跟输入通道走。"""

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
        self._alarm_code = [protocol.RAW_ANOMALY_OK] * self.channel_count
        self._hit_counts = [0] * self.channel_count
        self._clear_counts = [0] * self.channel_count

    def update(self, channel, raw_window):
        idx = int(channel)
        if idx < 0 or idx >= self.channel_count:
            return protocol.RAW_ANOMALY_OK
        if not self.enabled:
            return protocol.RAW_ANOMALY_OK

        code = detect_raw_window_anomaly(raw_window, self.cfg)
        if code != protocol.RAW_ANOMALY_OK:
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
            return protocol.RAW_ANOMALY_OK

        self._hit_counts[idx] = 0
        self._clear_counts[idx] += 1
        if self._clear_counts[idx] >= self.clear_count:
            self._alarm_on[idx] = False
            self._alarm_code[idx] = protocol.RAW_ANOMALY_OK
        if self._alarm_on[idx]:
            return self._alarm_code[idx]
        return protocol.RAW_ANOMALY_OK

    def codes(self):
        out = []
        for i in range(self.channel_count):
            if self._alarm_on[i]:
                out.append(int(self._alarm_code[i]))
            else:
                out.append(protocol.RAW_ANOMALY_OK)
        return out


def get_zero_guard_config(cfg):
    guard_cfg = cfg.get("zero_guard", None)
    if guard_cfg is None:
        preprocessing_cfg = cfg.get("preprocessing", {})
        guard_cfg = preprocessing_cfg.get("zero_guard", {})
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}
    return guard_cfg


def read_zero_guard_runtime_config(guard_cfg):
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


def zero_guard_percentile_from_sorted(values, pct):
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
        row = raw_seq if raw_ndim <= 1 else raw_seq[t]
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
    """按推理窗口顺序维护 0 干度保护的进入/退出滞回状态。"""

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


def update_zero_guard_from_freq_mean(freq_mean, guard_cfg, state):
    """在线路径只使用状态机真正依赖的频率均值，避免重复计算诊断特征。"""
    if not bool(guard_cfg.get("enabled", False)):
        return False, 0, {}
    runtime_cfg = read_zero_guard_runtime_config(guard_cfg)
    freq_mean = float(freq_mean)
    state.update(freq_mean)
    features = {
        "freq_mean": freq_mean,
        "zero_identity": freq_mean <= float(runtime_cfg["freq_enter_threshold"]),
        "zero_guard_state": bool(state.active),
        "enter_count": int(state.enter_count),
        "exit_count": int(state.exit_count),
    }
    return bool(state.active), int(state.enter_count), features


def compute_zero_guard_stateful_mask(raw_sequences, scaled_sequences=None, config=None):
    guard_cfg = config or {}
    runtime_cfg = read_zero_guard_runtime_config(guard_cfg)
    raw_shape = raw_sequences.shape
    raw_ndim = int(len(raw_shape))
    sample_count = 1 if raw_ndim <= 1 else int(raw_shape[0])

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
        raw_sample = raw_sequences if raw_ndim <= 1 else raw_sequences[i]
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
    """每个输出通道独立保存在线平滑状态。"""

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


class _LegacyFullGasAlarmState:
    """通用满气报警状态机。"""

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
            protocol.FULL_GAS_ALARM_CODE,
            self.history_size,
            self.danger_threshold,
            self.alarm_threshold,
            self.threshold_off,
            bool(self.alarm_on),
        )


class FullGasAlarmState:
    """Per-output high-dryness alarm state machine."""

    STATE_NORMAL = "NORMAL"
    STATE_OBSERVE = "OBSERVE"
    STATE_ALARM = "ALARM"

    def __init__(self, cfg, channel_count=1, output_names=None):
        if not isinstance(cfg, dict):
            cfg = {}
        self.enabled = bool(cfg.get("enabled", False))
        self.alarm_code = int(cfg.get("alarm_code", protocol.FULL_GAS_ALARM_CODE)) & 0xFF

        self.observe_prediction_count = clamp_count(cfg.get("observe_prediction_count", 6), 6, 1, 64)
        self.observe_prediction_hit = clamp_count(cfg.get("observe_prediction_hit", 4), 4, 1, self.observe_prediction_count)
        self.observe_prediction_min = float(cfg.get("observe_prediction_min", 0.62))
        self.observe_prediction_median = float(cfg.get("observe_prediction_median", 0.63))
        self.observe_frequency_count = clamp_count(cfg.get("observe_frequency_count", 6), 6, 1, 64)
        self.observe_frequency_hit = clamp_count(cfg.get("observe_frequency_hit", 4), 4, 1, self.observe_frequency_count)
        self.observe_frequency_threshold = float(cfg.get("observe_frequency_threshold", 529000.0))

        self.alarm_prediction_count = clamp_count(cfg.get("alarm_prediction_count", 3), 3, 1, 64)
        self.alarm_prediction_median = float(cfg.get("alarm_prediction_median", 0.67))
        self.alarm_frequency_count = clamp_count(cfg.get("alarm_frequency_count", 6), 6, 2, 64)
        self.alarm_frequency_hit = clamp_count(cfg.get("alarm_frequency_hit", 4), 4, 1, self.alarm_frequency_count)
        self.alarm_frequency_threshold = float(cfg.get("alarm_frequency_threshold", 529000.0))
        self.alarm_frequency_rise = float(cfg.get("alarm_frequency_rise", 30.0))

        self.hard_alarm_frequency_count = clamp_count(cfg.get("hard_alarm_frequency_count", 10), 10, 1, 64)
        self.hard_alarm_frequency_threshold = float(cfg.get("hard_alarm_frequency_threshold", 529200.0))
        self.hard_alarm_diff_count = clamp_count(cfg.get("hard_alarm_diff_count", 10), 10, 1, 64)
        self.hard_alarm_diff_hit = clamp_count(cfg.get("hard_alarm_diff_hit", 7), 7, 1, self.hard_alarm_diff_count)
        self.hard_alarm_diff_max = float(cfg.get("hard_alarm_diff_max", 58.0))

        self.clear_frequency_count = clamp_count(cfg.get("clear_frequency_count", 15), 15, 1, 64)
        self.clear_frequency_threshold = float(cfg.get("clear_frequency_threshold", 529000.0))
        self.clear_diff_count = clamp_count(cfg.get("clear_diff_count", 15), 15, 1, 64)
        self.clear_diff_hit = clamp_count(cfg.get("clear_diff_hit", 12), 12, 1, self.clear_diff_count)
        self.clear_diff_min = float(cfg.get("clear_diff_min", 60.0))
        self.clear_confirm_count = clamp_count(cfg.get("clear_confirm_count", 3), 3, 1, 64)

        self.clear_prediction_count = clamp_count(cfg.get("clear_prediction_count", 6), 6, 1, 64)
        self.clear_prediction_hit = clamp_count(cfg.get("clear_prediction_hit", 5), 5, 1, self.clear_prediction_count)
        self.clear_prediction_max = float(cfg.get("clear_prediction_max", 0.67))
        self.clear_prediction_frequency_count = clamp_count(cfg.get("clear_prediction_frequency_count", 10), 10, 1, 64)
        self.clear_prediction_frequency_threshold = float(cfg.get("clear_prediction_frequency_threshold", 529050.0))
        self.clear_prediction_diff_count = clamp_count(cfg.get("clear_prediction_diff_count", 10), 10, 1, 64)
        self.clear_prediction_diff_hit = clamp_count(
            cfg.get("clear_prediction_diff_hit", 7),
            7,
            1,
            self.clear_prediction_diff_count,
        )
        self.clear_prediction_diff_min = float(cfg.get("clear_prediction_diff_min", 58.0))

        self.normal_frequency_count = clamp_count(cfg.get("normal_frequency_count", 15), 15, 1, 64)
        self.normal_frequency_threshold = float(cfg.get("normal_frequency_threshold", 528900.0))
        self.normal_diff_count = clamp_count(cfg.get("normal_diff_count", 15), 15, 1, 64)
        self.normal_diff_hit = clamp_count(cfg.get("normal_diff_hit", 12), 12, 1, self.normal_diff_count)
        self.normal_diff_min = float(cfg.get("normal_diff_min", 60.0))
        self.normal_confirm_count = clamp_count(cfg.get("normal_confirm_count", 3), 3, 1, 64)

        self.history_size = max(
            self.observe_prediction_count,
            self.observe_frequency_count,
            self.alarm_prediction_count,
            self.alarm_frequency_count,
            self.hard_alarm_frequency_count,
            self.hard_alarm_diff_count,
            self.clear_frequency_count,
            self.clear_diff_count,
            self.clear_prediction_count,
            self.clear_prediction_frequency_count,
            self.clear_prediction_diff_count,
            self.normal_frequency_count,
            self.normal_diff_count,
        )

        self.channel_count = int(channel_count)
        if self.channel_count <= 0:
            self.channel_count = 1
        self.output_index_by_name = {}
        if output_names is not None:
            for i, name in enumerate(output_names):
                self.output_index_by_name[str(name)] = int(i)

        self.pred_history = [[] for _ in range(self.channel_count)]
        self.freq_history = [[] for _ in range(self.channel_count)]
        self.diff_history = [[] for _ in range(self.channel_count)]
        self.states = [self.STATE_NORMAL] * self.channel_count
        self.clear_hits = [0] * self.channel_count
        self.clear_prediction_hits = [0] * self.channel_count
        self.normal_hits = [0] * self.channel_count
        self.last_reason_by_channel = ["disabled" if not self.enabled else "warming"] * self.channel_count
        self.alarm_on = False
        self.last_reason = "disabled" if not self.enabled else "warming"

    def _append_history(self, history, value):
        history.append(float(value))
        while len(history) > self.history_size:
            del history[0]

    def _is_valid_prediction(self, value):
        try:
            if not protocol.is_finite_number(value):
                return False
        except Exception:
            return False
        value = float(value)
        return value >= 0.0 and value <= 1.0

    def _recent(self, history, count):
        count = int(count)
        if count <= 0 or len(history) < count:
            return None
        return history[len(history) - count :]

    def _median_recent(self, history, count):
        values = self._recent(history, count)
        if values is None:
            return None
        return median_list(values)

    def _count_recent_ge(self, history, count, threshold):
        values = self._recent(history, count)
        if values is None:
            return 0
        hits = 0
        for value in values:
            if float(value) >= float(threshold):
                hits += 1
        return hits

    def _count_recent_gt(self, history, count, threshold):
        values = self._recent(history, count)
        if values is None:
            return 0
        hits = 0
        for value in values:
            if float(value) > float(threshold):
                hits += 1
        return hits

    def _count_recent_le(self, history, count, threshold):
        values = self._recent(history, count)
        if values is None:
            return 0
        hits = 0
        for value in values:
            if float(value) <= float(threshold):
                hits += 1
        return hits

    def _all_recent_lt(self, history, count, threshold):
        values = self._recent(history, count)
        if values is None:
            return False
        for value in values:
            if not (float(value) < float(threshold)):
                return False
        return True

    def _all_recent_ge(self, history, count, threshold):
        values = self._recent(history, count)
        if values is None:
            return False
        for value in values:
            if not (float(value) >= float(threshold)):
                return False
        return True

    def _freq_rise_ok(self, history):
        values = self._recent(history, self.alarm_frequency_count)
        if values is None:
            return False
        split = int(len(values) // 2)
        if split <= 0:
            return False
        before = median_list(values[:split])
        after = median_list(values[len(values) - split :])
        return (after - before) >= self.alarm_frequency_rise

    def _observe_condition(self, idx):
        pred_median = self._median_recent(self.pred_history[idx], self.observe_prediction_count)
        pred_condition = (
            pred_median is not None
            and self._count_recent_ge(self.pred_history[idx], self.observe_prediction_count, self.observe_prediction_min)
            >= self.observe_prediction_hit
            and pred_median >= self.observe_prediction_median
        )
        freq_condition = (
            self._count_recent_ge(self.freq_history[idx], self.observe_frequency_count, self.observe_frequency_threshold)
            >= self.observe_frequency_hit
        )
        return pred_condition or freq_condition

    def _fused_alarm_condition(self, idx):
        pred_median = self._median_recent(self.pred_history[idx], self.alarm_prediction_count)
        if pred_median is None or pred_median < self.alarm_prediction_median:
            return False
        return (
            self._count_recent_ge(self.freq_history[idx], self.alarm_frequency_count, self.alarm_frequency_threshold)
            >= self.alarm_frequency_hit
            and self._freq_rise_ok(self.freq_history[idx])
        )

    def _hard_alarm_condition(self, idx):
        return (
            self._all_recent_ge(
                self.freq_history[idx],
                self.hard_alarm_frequency_count,
                self.hard_alarm_frequency_threshold,
            )
            and self._count_recent_le(self.diff_history[idx], self.hard_alarm_diff_count, self.hard_alarm_diff_max)
            >= self.hard_alarm_diff_hit
        )

    def _clear_frequency_condition(self, idx):
        return (
            self._all_recent_lt(self.freq_history[idx], self.clear_frequency_count, self.clear_frequency_threshold)
            and self._count_recent_gt(self.diff_history[idx], self.clear_diff_count, self.clear_diff_min)
            >= self.clear_diff_hit
        )

    def _clear_prediction_condition(self, idx):
        values = self._recent(self.pred_history[idx], self.clear_prediction_count)
        if values is None:
            return False
        pred_hits = 0
        for value in values:
            value = float(value)
            # A value below the former lower bound is stronger recovery evidence,
            # not a reason to keep a high-dryness alarm latched.
            if value <= self.clear_prediction_max:
                pred_hits += 1
        return (
            pred_hits >= self.clear_prediction_hit
            and self._all_recent_lt(
                self.freq_history[idx],
                self.clear_prediction_frequency_count,
                self.clear_prediction_frequency_threshold,
            )
            and self._count_recent_gt(
                self.diff_history[idx],
                self.clear_prediction_diff_count,
                self.clear_prediction_diff_min,
            )
            >= self.clear_prediction_diff_hit
        )

    def _normal_condition(self, idx):
        return (
            self._all_recent_lt(self.freq_history[idx], self.normal_frequency_count, self.normal_frequency_threshold)
            and self._count_recent_gt(self.diff_history[idx], self.normal_diff_count, self.normal_diff_min)
            >= self.normal_diff_hit
        )

    def _update_one(self, idx, prediction, frequency_mean, diff_std):
        if self._is_valid_prediction(prediction):
            self._append_history(self.pred_history[idx], prediction)
        self._append_history(self.freq_history[idx], frequency_mean)
        self._append_history(self.diff_history[idx], diff_std)

        state = self.states[idx]
        reason = "normal"
        if state == self.STATE_NORMAL:
            self.clear_hits[idx] = 0
            self.clear_prediction_hits[idx] = 0
            self.normal_hits[idx] = 0
            if self._observe_condition(idx):
                self.states[idx] = self.STATE_OBSERVE
                reason = "enter_observe"
        elif state == self.STATE_OBSERVE:
            self.clear_hits[idx] = 0
            self.clear_prediction_hits[idx] = 0
            if self._fused_alarm_condition(idx):
                self.states[idx] = self.STATE_ALARM
                self.normal_hits[idx] = 0
                reason = "fused"
            elif self._hard_alarm_condition(idx):
                self.states[idx] = self.STATE_ALARM
                self.normal_hits[idx] = 0
                reason = "hard_frequency"
            elif self._normal_condition(idx):
                self.normal_hits[idx] += 1
                reason = "observe_clear_count"
                if self.normal_hits[idx] >= self.normal_confirm_count:
                    self.states[idx] = self.STATE_NORMAL
                    self.normal_hits[idx] = 0
                    reason = "observe_clear"
            else:
                self.normal_hits[idx] = 0
                reason = "observe"
        else:
            self.normal_hits[idx] = 0
            if self._clear_frequency_condition(idx):
                self.clear_hits[idx] += 1
            else:
                self.clear_hits[idx] = 0
            if self._clear_prediction_condition(idx):
                self.clear_prediction_hits[idx] += 1
            else:
                self.clear_prediction_hits[idx] = 0
            reason = "alarm_hold"
            if self.clear_hits[idx] >= self.clear_confirm_count or self.clear_prediction_hits[idx] >= self.clear_confirm_count:
                self.states[idx] = self.STATE_OBSERVE
                self.clear_hits[idx] = 0
                self.clear_prediction_hits[idx] = 0
                reason = "alarm_clear"

        self.last_reason_by_channel[idx] = reason
        return self.states[idx] == self.STATE_ALARM

    def _reset_one_for_zero_guard(self, idx):
        """Clear only one output's alarm state after its zero guard is confirmed."""
        self.pred_history[idx] = []
        self.freq_history[idx] = []
        self.diff_history[idx] = []
        self.states[idx] = self.STATE_NORMAL
        self.clear_hits[idx] = 0
        self.clear_prediction_hits[idx] = 0
        self.normal_hits[idx] = 0
        self.last_reason_by_channel[idx] = "zero_guard_clear"

    def update(
        self,
        model_values,
        frequency_means,
        diff_stds,
        zero_guard_hit=False,
        zero_guard_hits=None,
    ):
        if not self.enabled:
            return None
        if not isinstance(frequency_means, (list, tuple)):
            frequency_means = [frequency_means] * len(model_values)
        if not isinstance(diff_stds, (list, tuple)):
            diff_stds = [diff_stds] * len(model_values)
        raw_zero_guard_hits = zero_guard_hits if zero_guard_hits is not None else zero_guard_hit
        if isinstance(raw_zero_guard_hits, (list, tuple)):
            channel_zero_guard_hits = raw_zero_guard_hits
        else:
            # Keep compatibility with the former scalar API. The online path now
            # always supplies one flag per model/input binding.
            channel_zero_guard_hits = [bool(raw_zero_guard_hits)] * len(model_values)
        limit = min(self.channel_count, len(model_values))
        for idx in range(limit):
            if idx < len(channel_zero_guard_hits) and bool(channel_zero_guard_hits[idx]):
                self._reset_one_for_zero_guard(idx)
                continue
            freq = frequency_means[idx] if idx < len(frequency_means) else 0.0
            diff_std = diff_stds[idx] if idx < len(diff_stds) else 0.0
            self._update_one(idx, model_values[idx], freq, diff_std)
        self.alarm_on = self.any_alarm()
        self.last_reason = self._combined_reason()
        return self.alarm_on

    def _combined_reason(self):
        for i in range(self.channel_count):
            if self.states[i] == self.STATE_ALARM:
                return "{}:{}".format(i, self.last_reason_by_channel[i])
        if self.channel_count > 0:
            return self.last_reason_by_channel[0]
        return "disabled" if not self.enabled else "normal"

    def any_alarm(self):
        for state in self.states:
            if state == self.STATE_ALARM:
                return True
        return False

    def is_alarm_index(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.channel_count:
            return False
        return self.states[idx] == self.STATE_ALARM

    def is_alarm_output(self, output_name):
        text = str(output_name)
        if text not in self.output_index_by_name:
            return False
        return self.is_alarm_index(self.output_index_by_name[text])

    def summary(self):
        if not self.enabled:
            return "disabled"
        return (
            "enabled=True, code=0x{:02X}, channels={}, observe_prediction_median={}, "
            "alarm_prediction_median={}, hard_alarm_frequency={}, any_alarm={}"
        ).format(
            self.alarm_code,
            self.channel_count,
            self.observe_prediction_median,
            self.alarm_prediction_median,
            self.hard_alarm_frequency_threshold,
            self.any_alarm(),
        )


def update_full_gas_alarm_state(
    alarm_state,
    model_values,
    frequency_means,
    diff_stds,
    zero_guard_hit=False,
    zero_guard_hits=None,
):
    if alarm_state is None or not alarm_state.enabled:
        return None
    return alarm_state.update(
        model_values,
        frequency_means,
        diff_stds,
        zero_guard_hit=zero_guard_hit,
        zero_guard_hits=zero_guard_hits,
    )
