import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(r"D:\code\network\VQ_Estimator\data\训练数据两次合集 - 去高干度")

SCALER_PATH = ROOT / "raw_cnn_k230/model/cnn-tcn/scaler_cnn_tcn_20260505_103801_u8u8_kld512.json"
scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
SCALER_MEAN = np.array(scaler["mean"], dtype=np.float64)
SCALER_SCALE = np.array(scaler.get("scale") or scaler.get("std"), dtype=np.float64)

THRESHOLD_SETS = {
    "current": (
        {"diff_p95_abs": 75.0, "win_range_mean": 280.0, "win_std_mean": 40.0, "absz_mean": 0.012},
        3,
    ),
    "proposal_A": (
        {"diff_p95_abs": 90.0, "win_range_mean": 300.0, "win_std_mean": 55.0, "absz_mean": 0.022},
        3,
    ),
    "proposal_B": (
        {"diff_p95_abs": 90.0, "win_range_mean": 300.0, "win_std_mean": 52.0, "absz_mean": 0.0215},
        3,
    ),
    "strict4": (
        {"diff_p95_abs": 90.0, "win_range_mean": 300.0, "win_std_mean": 55.0, "absz_mean": 0.022},
        4,
    ),
}


def label_from_name(name):
    match = re.match(r"([0-9]+(?:\.[0-9]+)?)", name)
    return float(match.group(1)) if match else math.nan


def read_csv_values(path):
    values = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for part in re.split(r"[,\s]+", line.strip()):
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                pass
    return np.array(values, dtype=np.float64)


def iter_features(values):
    windows = [values[start : start + 500] for start in range(0, len(values) - 500 + 1, 200)]
    for win_idx in range(4, len(windows)):
        raw = np.stack(windows[win_idx - 4 : win_idx + 1])
        demean = raw - raw.mean(axis=1, keepdims=True)
        scaled = (demean - SCALER_MEAN) / SCALER_SCALE
        diffs = np.abs(np.diff(raw, axis=1)).reshape(-1)
        yield {
            "diff_p95_abs": float(np.percentile(diffs, 95)),
            "win_range_mean": float((raw.max(axis=1) - raw.min(axis=1)).mean()),
            "win_std_mean": float(raw.std(axis=1).mean()),
            "absz_mean": float(np.abs(scaled).mean()),
        }


def scan():
    rows = []
    for path in sorted(FOLDER.glob("*.csv")):
        features = list(iter_features(read_csv_values(path)))
        row = {"name": path.name, "label": label_from_name(path.name), "triggers": len(features)}
        for key, (thresholds, min_votes) in THRESHOLD_SETS.items():
            hits = []
            votes = []
            for feature in features:
                vote_count = sum(feature[name] <= thresholds[name] for name in thresholds)
                votes.append(vote_count)
                hits.append(vote_count >= min_votes)
            row[key] = sum(hits) / len(hits) if hits else 0.0
            row[key + "_maxvote"] = max(votes) if votes else 0
        rows.append(row)
    return rows


def print_summary(rows):
    for key in THRESHOLD_SETS:
        print("\n== {} ==".format(key))
        for low, high in [(0, 0.0001), (0.0001, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0)]:
            group = [row for row in rows if low <= row["label"] < high]
            if not group:
                continue
            triggers = sum(row["triggers"] for row in group)
            hit_frames = sum(row[key] * row["triggers"] for row in group)
            hit_files = sum(1 for row in group if row[key] > 0)
            print(
                "label[{},{}) files={} files_hit={} frame_hit_rate={:.4f}".format(
                    low, high, len(group), hit_files, hit_frames / triggers
                )
            )
        bad = [row for row in rows if row["label"] > 0.0001 and row[key] > 0]
        print("nonzero_files_hit", len(bad))
        for row in bad[:25]:
            print(
                " {} label={} hit={:.4f} triggers={} maxvote={}".format(
                    row["name"], row["label"], row[key], row["triggers"], row[key + "_maxvote"]
                )
            )


def write_csv(rows):
    out = ROOT / "data/vq_zero_guard_threshold_scan.csv"
    header = [
        "name",
        "label",
        "triggers",
        "current",
        "current_maxvote",
        "proposal_A",
        "proposal_A_maxvote",
        "proposal_B",
        "proposal_B_maxvote",
        "strict4",
        "strict4_maxvote",
    ]
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    [
                        row["name"],
                        str(row["label"]),
                        str(row["triggers"]),
                        "{:.6f}".format(row["current"]),
                        str(row["current_maxvote"]),
                        "{:.6f}".format(row["proposal_A"]),
                        str(row["proposal_A_maxvote"]),
                        "{:.6f}".format(row["proposal_B"]),
                        str(row["proposal_B_maxvote"]),
                        "{:.6f}".format(row["strict4"]),
                        str(row["strict4_maxvote"]),
                    ]
                )
                + "\n"
            )
    print("\nwrote", out)


if __name__ == "__main__":
    result_rows = scan()
    print_summary(result_rows)
    write_csv(result_rows)
