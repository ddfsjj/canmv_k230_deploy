import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raw_cnn_k230"))

from runtime import guards  # noqa: E402


SCALER_PATH = ROOT / "raw_cnn_k230/model/cnn-tcn/scaler_cnn_tcn_20260505_103801_u8u8_kld512.json"
CONFIG_PATH = ROOT / "raw_cnn_k230/configs/runtime.json"


def read_scaler():
    scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    return (
        np.array(scaler["mean"], dtype=np.float64),
        np.array(scaler.get("scale") or scaler.get("std"), dtype=np.float64),
    )


def read_guard_config():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "status" in cfg:
        return cfg.get("status", {}).get("zero_guard", {})
    return cfg["zero_guard"]


def extract_last_20_numbers(path):
    values = []
    bad_lines = 0
    first_rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        numbers = []
        for token in line.strip().split():
            if token == "??":
                break
            try:
                numbers.append(int(token))
            except ValueError:
                pass
        if len(numbers) < 20:
            bad_lines += 1
            continue
        row = numbers[-20:]
        if len(first_rows) < 3:
            first_rows.append(row)
        values.extend(row)
    return np.array(values, dtype=np.float64), bad_lines, first_rows


def compute_features(raw_seq, mean, scale):
    demean = raw_seq - raw_seq.mean(axis=1, keepdims=True)
    scaled = (demean - mean) / scale
    diffs = np.abs(np.diff(raw_seq, axis=1)).reshape(-1)
    return {
        "diff_p95_abs": float(np.percentile(diffs, 95)),
        "win_range_mean": float((raw_seq.max(axis=1) - raw_seq.min(axis=1)).mean()),
        "win_std_mean": float(raw_seq.std(axis=1).mean()),
        "freq_mean": float(raw_seq.reshape(-1).mean()),
        "absz_mean": float(np.abs(scaled).mean()),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/check_zero_guard_txt.py <txt/log path>")
    path = Path(sys.argv[1])
    mean, scale = read_scaler()
    guard = read_guard_config()
    values, bad_lines, first_rows = extract_last_20_numbers(path)

    windows = [values[start : start + 500] for start in range(0, len(values) - 500 + 1, 200)]
    rows = []
    state = guards.ZeroGuardState(guard)
    for win_idx in range(4, len(windows)):
        raw_seq = np.stack(windows[win_idx - 4 : win_idx + 1])
        features = compute_features(raw_seq, mean, scale)
        zero_identity = features["freq_mean"] <= float(guard.get("freq_enter_threshold", 480000.0))
        hit = state.update(features["freq_mean"]) if bool(guard.get("enabled", False)) else False
        features["zero_identity"] = zero_identity
        features["zero_guard_state"] = bool(state.active)
        features["enter_count"] = int(state.enter_count)
        features["exit_count"] = int(state.exit_count)
        rows.append(
            {
                "rx": win_idx * 200 + 500,
                "features": features,
                "hit": bool(hit),
            }
        )

    print("file:", path)
    print("source_lines:", len(path.read_text(encoding="utf-8", errors="ignore").splitlines()))
    print("extracted_values:", len(values))
    print("bad_lines:", bad_lines)
    print("first_rows:", first_rows)
    if len(values):
        print(
            "signal_stats:",
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            },
        )
    print(
        "zero_guard:",
        {
            "enabled": bool(guard.get("enabled", False)),
            "freq_enter_threshold": float(guard.get("freq_enter_threshold", 480000.0)),
            "freq_exit_threshold": float(guard.get("freq_exit_threshold", 500000.0)),
            "enter_consecutive_windows": int(guard.get("enter_consecutive_windows", 3)),
            "exit_consecutive_windows": int(guard.get("exit_consecutive_windows", 3)),
        },
    )
    print("triggers:", len(rows))
    if rows:
        hit_rate = sum(row["hit"] for row in rows) / float(len(rows))
        print("hit_rate:", hit_rate)
        print("first_12:")
        for row in rows[:12]:
            print(
                {
                    "rx": row["rx"],
                    "hit": bool(row["hit"]),
                    "features": {key: round(value, 6) for key, value in row["features"].items()},
                }
            )
        for key in ("freq_mean", "diff_p95_abs", "win_range_mean", "win_std_mean", "absz_mean"):
            vals = np.array([row["features"][key] for row in rows], dtype=np.float64)
            print(
                key,
                {
                    "min": round(float(vals.min()), 6),
                    "p50": round(float(np.percentile(vals, 50)), 6),
                    "p90": round(float(np.percentile(vals, 90)), 6),
                    "p95": round(float(np.percentile(vals, 95)), 6),
                    "max": round(float(vals.max()), 6),
                },
            )


if __name__ == "__main__":
    main()
