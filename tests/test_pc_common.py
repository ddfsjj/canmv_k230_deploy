import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raw_cnn_pc"))
sys.path.insert(0, str(ROOT / "raw_cnn_k230"))

from raw_cnn import data as pc_data  # noqa: E402
from runtime import features as k230_features  # noqa: E402

try:
    import torch  # noqa: E402
    from raw_cnn import models as pc_models  # noqa: E402
except Exception as exc:  # pragma: no cover - 只在缺少 torch 时触发
    torch = None
    pc_models = None
    MODEL_IMPORT_ERROR = exc


class PcCommonDataTest(unittest.TestCase):
    def test_feature_modes_match_k230_runtime(self):
        window = np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float32)
        for mode in ("raw", "window_demean", "window_rel_demean"):
            pc_out = pc_data.apply_feature_mode(window, mode)
            k230_out = np.empty_like(window)
            k230_features.apply_feature_mode_1d(window, mode, k230_out)
            np.testing.assert_allclose(pc_out, k230_out, rtol=1e-6, atol=1e-6)

    def test_build_dataset_keeps_sorted_csv_and_window_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "0.2-b.csv").write_text("\n".join(str(i) for i in range(1, 8)), encoding="utf-8")
            (data_dir / "0.1-a.csv").write_text("\n".join(str(i) for i in range(10, 17)), encoding="utf-8")

            X, y = pc_data.build_dataset(
                data_dir=data_dir,
                base_window_size=3,
                base_step=2,
                seq_length=2,
                seq_step=1,
                feature_mode="raw",
            )

        self.assertEqual(X.shape, (4, 2, 3))
        np.testing.assert_allclose(y, np.asarray([0.1, 0.1, 0.2, 0.2], dtype=np.float32))
        np.testing.assert_allclose(X[0], np.asarray([[10, 11, 12], [12, 13, 14]], dtype=np.float32))
        np.testing.assert_allclose(X[2], np.asarray([[1, 2, 3], [3, 4, 5]], dtype=np.float32))


@unittest.skipIf(pc_models is None, "torch/raw_cnn.models unavailable")
class PcCommonModelTest(unittest.TestCase):
    def test_model_type_aliases_are_stable(self):
        self.assertEqual(pc_models.normalize_model_type("CNN"), "cnn_all")
        self.assertEqual(pc_models.normalize_model_type("CNN-LSTM"), "cnn_lstm")
        self.assertEqual(pc_models.normalize_model_type("CNN TCN"), "cnn_tcn")
        self.assertEqual(pc_models.normalize_model_type("cnn_tcn_seg3"), "cnn_tcn_seg3_soft_stats_moe")

    def test_cnn_tcn_builds_and_runs(self):
        model = pc_models.build_model_from_config(
            {
                "type": "CNN-TCN",
                "cnn_tcn_conv_filters": [4, 6],
                "cnn_tcn_kernel_size": [3, 3],
                "cnn_tcn_pool_size": [2, 2],
                "cnn_tcn_num_channels": [5],
                "cnn_tcn_tcn_kernel_size": 3,
                "cnn_tcn_dilations": [1],
                "cnn_tcn_dropout": 0.0,
            },
            input_shape=(3, 16),
        )
        with torch.no_grad():
            out = model(torch.zeros((2, 3, 16), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_cnn_lstm_uses_state_dict_layout_when_available(self):
        model = pc_models.CNNLSTM(
            input_shape=(2, 16),
            conv_filters=[4],
            kernel_size=3,
            pool_size=2,
            lstm_hidden_size=7,
            lstm_num_layers=2,
            lstm_bidirectional=True,
        )
        rebuilt = pc_models.build_model_from_config(
            {
                "type": "CNN-LSTM",
                "conv_filters": [4],
                "kernel_size": 3,
                "pool_size": 2,
                "lstm_hidden_size": 99,
                "lstm_num_layers": 1,
                "lstm_bidirectional": False,
            },
            input_shape=(2, 16),
            state_dict=model.state_dict(),
        )
        self.assertEqual(rebuilt.lstm.hidden_size, 7)
        self.assertEqual(rebuilt.lstm.num_layers, 2)
        self.assertTrue(rebuilt.lstm.bidirectional)


if __name__ == "__main__":
    unittest.main()
