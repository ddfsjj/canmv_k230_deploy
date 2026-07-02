import argparse
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_runtime_config_from_export as updater  # noqa: E402


def make_args(**overrides):
    values = {
        "append": False,
        "model_index": 0,
        "model_name": None,
        "output_name": None,
        "input_channels": None,
        "output_slots": None,
        "profile_name": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class UpdateRuntimeConfigFromExportTest(unittest.TestCase):
    def make_export(self):
        return {
            "name": "exported_cnn_tcn",
            "paths": {
                "kmodel": "../raw_cnn_k230/model/cnn-tcn/new.kmodel",
                "scaler_json": "../raw_cnn_k230/model/cnn-tcn/new_scaler.json",
            },
            "data": {
                "base_window_size": 500,
                "base_step": 200,
                "sequence_length": 5,
                "sequence_step": 1,
            },
            "preprocessing": {"feature_mode": "window_demean"},
            "model": {"type": "CNN-TCN"},
        }

    def make_runtime(self):
        return {
            "version": 1,
            "profile_name": "runtime",
            "models": [
                {
                    "name": "old_model",
                    "enabled": True,
                    "model_type": "cnn_tcn",
                    "input_channels": [0],
                    "output": {"name": "old_model", "slots": {"0": 2}, "scale": 100},
                    "assets": {"kmodel": "model/old.kmodel", "scaler_json": "model/old.json"},
                    "window": {"base_window_size": 100, "base_step": 50, "sequence_length": 1, "sequence_step": 1},
                }
            ],
        }

    def test_replaces_existing_model_preserving_io_when_not_overridden(self):
        updated = updater.update_runtime_config(self.make_runtime(), self.make_export(), make_args())
        model = updated["models"][0]
        self.assertEqual(model["name"], "old_model")
        self.assertEqual(model["model_type"], "cnn_tcn")
        self.assertEqual(model["input_channels"], [0])
        self.assertEqual(model["output"]["slots"], {"0": 2})
        self.assertEqual(model["assets"]["kmodel"], "model/cnn-tcn/new.kmodel")
        self.assertEqual(model["assets"]["scaler_json"], "model/cnn-tcn/new_scaler.json")
        self.assertEqual(model["window"]["sequence_length"], 5)
        self.assertEqual(model["window"]["feature_mode"], "window_demean")

    def test_overrides_channels_slots_and_names(self):
        updated = updater.update_runtime_config(
            self.make_runtime(),
            self.make_export(),
            make_args(model_name="new_name", output_name="new_output", input_channels="0,1", output_slots="0:0,1:1"),
        )
        model = updated["models"][0]
        self.assertEqual(model["name"], "new_name")
        self.assertEqual(model["output"]["name"], "new_output")
        self.assertEqual(model["input_channels"], [0, 1])
        self.assertEqual(model["output"]["slots"], {"0": 0, "1": 1})

    def test_append_adds_model(self):
        updated = updater.update_runtime_config(self.make_runtime(), self.make_export(), make_args(append=True))
        self.assertEqual(len(updated["models"]), 2)
        self.assertEqual(updated["models"][1]["name"], "exported_cnn_tcn")

    def test_script_output_can_be_parsed_as_json(self):
        payload = updater.update_runtime_config(self.make_runtime(), self.make_export(), make_args())
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self.assertEqual(json.loads(text)["models"][0]["assets"]["kmodel"], "model/cnn-tcn/new.kmodel")


if __name__ == "__main__":
    unittest.main()
