import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = ROOT / "raw_cnn_k230/model/cnn-tcn/scaler_cnn_tcn_20260505_103801_u8u8_kld512.json"
CONFIG_PATH = ROOT / "raw_cnn_k230/configs/k230_config_multi.json"


def read_scaler():
    scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    return (
        np.array(scaler["mean"], dtype=np.float64),
        np.array(scaler.get("scale") or scaler.get("std"), dtype=np.float64),
    )


def read_guard_config():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    guard = cfg["zero_guard"]
    return guard["thresholds"], int(guard.get("min_votes", 3))


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
        "absz_mean": float(np.abs(scaled).mean()),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/check_zero_guard_txt.py <txt/log path>")
    path = Path(sys.argv[1])
    mean, scale = read_scaler()
    thresholds, min_votes = read_guard_config()
    values, bad_lines, first_rows = extract_last_20_numbers(path)

    windows = [values[start : start + 500] for start in range(0, len(values) - 500 + 1, 200)]
    rows = []
    for win_idx in range(4, len(windows)):
        raw_seq = np.stack(windows[win_idx - 4 : win_idx + 1])
        features = compute_features(raw_seq, mean, scale)
        votes = sum(features[key] <= thresholds[key] for key in thresholds)
        rows.append(
            {
                "rx": win_idx * 200 + 500,
                "features": features,
                "votes": votes,
                "hit": votes >= min_votes,
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
    print("thresholds:", thresholds)
    print("min_votes:", min_votes)
    print("triggers:", len(rows))
    if rows:
        hit_rate = sum(row["hit"] for row in rows) / float(len(rows))
        print("hit_rate:", hit_rate)
        print("first_12:")
        for row in rows[:12]:
            print(
                {
                    "rx": row["rx"],
                    "votes": int(row["votes"]),
                    "hit": bool(row["hit"]),
                    "features": {key: round(value, 6) for key, value in row["features"].items()},
                }
            )
        for key in thresholds:
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
