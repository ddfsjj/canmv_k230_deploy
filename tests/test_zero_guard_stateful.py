import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raw_cnn_k230"))

from runtime import guards  # noqa: E402


def make_windows(freq_values, width=8):
    return np.asarray([[float(freq)] * int(width) for freq in freq_values], dtype=np.float32)


class ZeroGuardStatefulTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "enabled": True,
            "freq_enter_threshold": 480000.0,
            "freq_exit_threshold": 500000.0,
            "enter_consecutive_windows": 3,
            "exit_consecutive_windows": 3,
            "confidence_absz_threshold": 3.0,
        }

    def test_enters_after_three_low_frequency_windows(self):
        mask, features, _ = guards.compute_zero_guard_stateful_mask(
            make_windows([479000.0, 478000.0, 477000.0]),
            config=self.cfg,
        )
        self.assertEqual(mask, [False, False, True])
        self.assertEqual(features["zero_identity"], [True, True, True])

    def test_one_or_two_low_frequency_windows_do_not_enter(self):
        mask, _, _ = guards.compute_zero_guard_stateful_mask(
            make_windows([479000.0, 478000.0]),
            config=self.cfg,
        )
        self.assertEqual(mask, [False, False])

    def test_exits_only_after_three_high_frequency_windows(self):
        mask, features, _ = guards.compute_zero_guard_stateful_mask(
            make_windows([479000.0, 478000.0, 477000.0, 490000.0, 501000.0, 502000.0, 503000.0]),
            config=self.cfg,
        )
        self.assertEqual(mask, [False, False, True, True, True, True, False])
        self.assertEqual(features["exit_count"][-3:], [1, 2, 0])

    def test_high_dryness_low_variation_sample_does_not_enter(self):
        raw = make_windows([527000.0, 527000.0, 527000.0, 527000.0])
        mask, features, _ = guards.compute_zero_guard_stateful_mask(raw, config=self.cfg)
        self.assertEqual(mask, [False, False, False, False])
        self.assertTrue(all(value <= 0.001 for value in features["win_std_mean"]))
        self.assertTrue(all(value <= 0.001 for value in features["diff_p95_abs"]))
        self.assertEqual(features["zero_identity"], [False, False, False, False])

    def test_old_config_shape_is_accepted(self):
        old_cfg = {
            "enabled": True,
            "output_value": 0.0,
            "min_votes": 3,
            "thresholds": {
                "diff_p95_abs": 90.0,
                "win_range_mean": 300.0,
                "win_std_mean": 55.0,
                "absz_mean": 0.022,
            },
        }
        mask, features, _ = guards.compute_zero_guard_stateful_mask(
            make_windows([479000.0, 478000.0, 477000.0]),
            config=old_cfg,
        )
        self.assertEqual(mask, [False, False, True])
        self.assertIn("freq_mean", features)

    def test_fast_online_check_preserves_state_transitions(self):
        state = guards.ZeroGuardState(self.cfg)
        results = [
            guards.update_zero_guard_from_freq_mean(value, self.cfg, state)[0]
            for value in [479000.0, 478000.0, 477000.0, 501000.0, 502000.0, 503000.0]
        ]
        self.assertEqual(results, [False, False, True, True, True, False])


if __name__ == "__main__":
    unittest.main()
