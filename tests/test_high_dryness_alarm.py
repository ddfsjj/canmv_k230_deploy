import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raw_cnn_k230"))

from runtime import guards  # noqa: E402
from runtime import output as runtime_output  # noqa: E402


def alarm_cfg():
    return {
        "enabled": True,
        "alarm_code": 0x33,
        "observe_prediction_count": 6,
        "observe_prediction_hit": 4,
        "observe_prediction_min": 0.62,
        "observe_prediction_median": 0.63,
        "observe_frequency_count": 6,
        "observe_frequency_hit": 4,
        "observe_frequency_threshold": 529000,
        "alarm_prediction_count": 3,
        "alarm_prediction_median": 0.67,
        "alarm_frequency_count": 6,
        "alarm_frequency_hit": 4,
        "alarm_frequency_threshold": 529000,
        "alarm_frequency_rise": 30,
        "hard_alarm_frequency_count": 10,
        "hard_alarm_frequency_threshold": 529200,
        "hard_alarm_diff_count": 10,
        "hard_alarm_diff_hit": 7,
        "hard_alarm_diff_max": 58,
        "clear_frequency_count": 15,
        "clear_frequency_threshold": 529000,
        "clear_diff_count": 15,
        "clear_diff_hit": 12,
        "clear_diff_min": 60,
        "clear_confirm_count": 3,
        "clear_prediction_count": 6,
        "clear_prediction_hit": 5,
        "clear_prediction_max": 0.67,
        "clear_prediction_frequency_count": 10,
        "clear_prediction_frequency_threshold": 529050,
        "clear_prediction_diff_count": 10,
        "clear_prediction_diff_hit": 7,
        "clear_prediction_diff_min": 58,
        "normal_frequency_count": 15,
        "normal_frequency_threshold": 528900,
        "normal_diff_count": 15,
        "normal_diff_hit": 12,
        "normal_diff_min": 60,
        "normal_confirm_count": 3,
    }


class HighDrynessAlarmTest(unittest.TestCase):
    @staticmethod
    def _trigger_fused_alarm(state, alarm_index=0, channel_count=1):
        freqs = [528980, 529000, 529010, 529020, 529060, 529070, 529080]
        preds = [0.63, 0.63, 0.63, 0.63, 0.67, 0.67, 0.67]
        for pred, freq in zip(preds, freqs):
            values = [0.30] * channel_count
            frequency_means = [528000] * channel_count
            diff_stds = [70] * channel_count
            values[alarm_index] = pred
            frequency_means[alarm_index] = freq
            state.update(values, frequency_means, diff_stds)

    @staticmethod
    def _trigger_hard_alarm(state, alarm_index=0, channel_count=1):
        for _ in range(10):
            values = [0.01] * channel_count
            frequency_means = [528000] * channel_count
            diff_stds = [70] * channel_count
            frequency_means[alarm_index] = 529250
            diff_stds[alarm_index] = 20
            state.update(values, frequency_means, diff_stds)

    def test_fused_alarm_is_per_output_and_raw_error_has_priority(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=2, output_names=["m_ch0", "m_ch1"])
        freqs = [529000, 529010, 529020, 529060, 529070, 529080, 529100]
        preds = [0.64, 0.64, 0.64, 0.64, 0.68, 0.68, 0.68]
        for pred, freq in zip(preds, freqs):
            state.update([pred, 0.30], [freq, 528000], [70, 70])

        self.assertTrue(state.is_alarm_output("m_ch0"))
        self.assertFalse(state.is_alarm_output("m_ch1"))

        output_cfg = {"slots": ["m_ch0", "m_ch1"], "fill_value": 0.0}
        codes = runtime_output.build_slot_error_codes(
            output_cfg,
            {"m_ch0": 0, "m_ch1": 1},
            [0, 0],
            state,
            2,
        )
        self.assertEqual(codes, [0x33, 0])

        codes = runtime_output.build_slot_error_codes(
            output_cfg,
            {"m_ch0": 0, "m_ch1": 1},
            [0x01, 0],
            state,
            2,
        )
        self.assertEqual(codes, [0x01, 0])

    def test_hard_frequency_alarm_does_not_need_prediction(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        self._trigger_hard_alarm(state)
        self.assertTrue(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.last_reason, "0:hard_frequency")

    def test_twelve_outputs_are_isolated(self):
        names = ["m_ch{}".format(i) for i in range(12)]
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=12, output_names=names)
        self._trigger_fused_alarm(state, alarm_index=5, channel_count=12)

        alarm_indices = [i for i, name in enumerate(names) if state.is_alarm_output(name)]
        self.assertEqual(alarm_indices, [5])

    def test_frequency_recovery_clears_after_required_confirmation(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        self._trigger_hard_alarm(state)

        for _ in range(16):
            state.update([0.80], [528000], [70])
        self.assertTrue(state.is_alarm_output("m_ch0"))

        state.update([0.80], [528000], [70])
        self.assertFalse(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.states[0], guards.FullGasAlarmState.STATE_OBSERVE)

    def test_prediction_recovery_clears_after_required_confirmation(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        self._trigger_fused_alarm(state)

        for _ in range(11):
            state.update([0.63], [528950], [70])
        self.assertTrue(state.is_alarm_output("m_ch0"))

        state.update([0.63], [528950], [70])
        self.assertFalse(state.is_alarm_output("m_ch0"))

    def test_predictions_below_former_lower_bound_also_clear_alarm(self):
        for recovered_prediction in (0.30, 0.40, 0.50):
            state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
            self._trigger_fused_alarm(state)

            for _ in range(11):
                state.update([recovered_prediction], [528950], [70])
            self.assertTrue(state.is_alarm_output("m_ch0"))

            state.update([recovered_prediction], [528950], [70])
            self.assertFalse(state.is_alarm_output("m_ch0"))

    def test_prediction_above_clear_max_does_not_use_prediction_clear_path(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        self._trigger_fused_alarm(state)

        for _ in range(12):
            state.update([0.671], [528950], [70])

        self.assertTrue(state.is_alarm_output("m_ch0"))

    def test_exact_fused_thresholds_trigger_alarm(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        preds = [0.63, 0.63, 0.63, 0.63, 0.67, 0.67, 0.67]
        freqs = [528000, 528990, 528990, 529000, 529020, 529020, 529020]
        for pred, freq in zip(preds, freqs):
            state.update([pred], [freq], [70])

        self.assertTrue(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.last_reason, "0:fused")

    def test_prediction_just_below_fused_threshold_does_not_alarm(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        preds = [0.63, 0.63, 0.63, 0.63, 0.669, 0.669, 0.669]
        freqs = [528000, 528990, 528990, 529000, 529020, 529020, 529020]
        for pred, freq in zip(preds, freqs):
            state.update([pred], [freq], [70])

        self.assertFalse(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.states[0], guards.FullGasAlarmState.STATE_OBSERVE)

    def test_exact_hard_frequency_threshold_triggers_alarm(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        for _ in range(10):
            state.update([0.01], [529200], [58])

        self.assertTrue(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.last_reason, "0:hard_frequency")

    def test_zero_guard_clears_alarm_and_history(self):
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=1, output_names=["m_ch0"])
        self._trigger_hard_alarm(state)

        state.update([0.0], [470000], [20], zero_guard_hits=[True])

        self.assertFalse(state.is_alarm_output("m_ch0"))
        self.assertEqual(state.states[0], guards.FullGasAlarmState.STATE_NORMAL)
        self.assertEqual(state.last_reason, "zero_guard_clear")
        self.assertEqual(state.pred_history[0], [])
        self.assertEqual(state.freq_history[0], [])
        self.assertEqual(state.diff_history[0], [])

    def test_zero_guard_only_clears_its_own_output_and_can_rearm(self):
        names = ["m_ch{}".format(i) for i in range(12)]
        state = guards.FullGasAlarmState(alarm_cfg(), channel_count=12, output_names=names)
        self._trigger_hard_alarm(state, alarm_index=3, channel_count=12)
        self._trigger_hard_alarm(state, alarm_index=7, channel_count=12)
        self.assertTrue(state.is_alarm_output("m_ch3"))
        self.assertTrue(state.is_alarm_output("m_ch7"))

        zero_guard_hits = [False] * 12
        zero_guard_hits[3] = True
        state.update([0.0] * 12, [470000] * 12, [20] * 12, zero_guard_hits=zero_guard_hits)

        self.assertFalse(state.is_alarm_output("m_ch3"))
        self.assertTrue(state.is_alarm_output("m_ch7"))
        self.assertEqual(state.last_reason_by_channel[3], "zero_guard_clear")

        self._trigger_fused_alarm(state, alarm_index=3, channel_count=12)
        self.assertTrue(state.is_alarm_output("m_ch3"))
        self.assertTrue(state.is_alarm_output("m_ch7"))


if __name__ == "__main__":
    unittest.main()
