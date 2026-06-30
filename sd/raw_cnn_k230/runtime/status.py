"""统一状态层适配。"""

from runtime import protocol


class RuntimeStatusContext:
    """中文注释：集中持有在线推理所需的状态对象和状态配置。"""

    def __init__(self, base_module, cfg, runtime_cfg, channel_count, binding_count):
        self.base = base_module
        self.zero_guard_cfg = base_module.get_zero_guard_config(cfg)
        self.zero_guard_enabled = bool(self.zero_guard_cfg.get("enabled", False))
        self.zero_guard_output_value = float(self.zero_guard_cfg.get("output_value", 0.0))
        self.raw_anomaly = base_module.RawChannelAnomalyState(
            base_module.get_raw_anomaly_config(cfg),
            channel_count,
        )
        self.channel_raw_error_codes = [protocol.RAW_ANOMALY_OK] * int(channel_count)
        self.postprocessor = base_module.create_runtime_postprocessor(
            cfg,
            channel_count=int(binding_count),
        )
        alarm_cfg = base_module.get_runtime_section(runtime_cfg, "full_gas_alarm")
        self.full_gas_alarm = base_module.FullGasAlarmState(alarm_cfg)

    def update_raw_anomaly(self, source_index, raw_window):
        """中文注释：更新某个物理输入通道的原始异常码。"""
        code = self.raw_anomaly.update(source_index, raw_window)
        self.channel_raw_error_codes[int(source_index)] = code
        return code

    def update_full_gas_alarm(self, model_values, freq_mean, zero_guard_hit=False):
        """中文注释：按统一入口更新满液报警状态。"""
        if self.full_gas_alarm.enabled:
            self.full_gas_alarm.update(
                model_values,
                freq_mean,
                zero_guard_hit=zero_guard_hit,
            )

    def postprocessing_summary(self):
        return "enabled={}, type={}".format(
            bool(self.postprocessor.enabled),
            self.postprocessor.kind,
        )

    def raw_anomaly_summary(self):
        return "enabled={}, hit_count={}, clear_count={}".format(
            bool(self.raw_anomaly.enabled),
            int(self.raw_anomaly.hit_count),
            int(self.raw_anomaly.clear_count),
        )

    def zero_guard_summary(self):
        return "enabled={}, output_value={}, enter_freq={}, exit_freq={}, enter_windows={}, exit_windows={}".format(
            bool(self.zero_guard_enabled),
            self.zero_guard_output_value,
            float(self.zero_guard_cfg.get("freq_enter_threshold", 480000.0)),
            float(self.zero_guard_cfg.get("freq_exit_threshold", 500000.0)),
            int(self.zero_guard_cfg.get("enter_consecutive_windows", 3)),
            int(self.zero_guard_cfg.get("exit_consecutive_windows", 3)),
        )

    def full_gas_summary(self):
        return self.full_gas_alarm.summary()


def format_zero_guard_features(features):
    """中文注释：统一 zero guard 日志里的特征格式。"""
    return {
        "freq_mean": round(float(features.get("freq_mean", 0.0)), 3),
        "diff_p95_abs": round(float(features.get("diff_p95_abs", 0.0)), 3),
        "win_range_mean": round(float(features.get("win_range_mean", 0.0)), 3),
        "win_std_mean": round(float(features.get("win_std_mean", 0.0)), 3),
        "absz_mean": round(float(features.get("absz_mean", 0.0)), 6),
        "zero_identity": bool(features.get("zero_identity", False)),
        "zero_confidence_high": bool(features.get("zero_confidence_high", False)),
        "zero_guard_state": bool(features.get("zero_guard_state", False)),
        "enter_count": int(features.get("enter_count", 0)),
        "exit_count": int(features.get("exit_count", 0)),
    }


def check_zero_guard(base_module, raw_seq, scaled_seq, guard_cfg, state=None):
    """中文注释：统一 zero guard 判定入口。"""
    return base_module.is_zero_guard_hit(raw_seq, scaled_seq, guard_cfg, state=state)


def build_raw_anomaly_state(base_module, cfg, channel_count):
    """中文注释：创建原始通道异常状态机。"""
    return base_module.RawChannelAnomalyState(cfg, channel_count)


def build_postprocessor(base_module, cfg, channel_count):
    """中文注释：创建统一后处理器。"""
    return base_module.create_runtime_postprocessor(cfg, channel_count=channel_count)
