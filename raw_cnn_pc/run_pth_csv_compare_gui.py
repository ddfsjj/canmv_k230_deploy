import csv
import json
import threading
import time
import traceback
from pathlib import Path
from tkinter import END, Button, Entry, Label, Radiobutton, StringVar, Text, Tk, filedialog, messagebox

import joblib
import numpy as np
import torch

import infer


DEFAULT_INFER_CONFIG = "configs/infer/infer_config_cnn_tcn.json"
DEFAULT_OUTPUT_DIR = "artifacts/pth_compare"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_rows_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def resolve_under_root(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def model_tag(path: Path) -> str:
    name = path.stem if path.is_file() else path.name
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def find_one_file(model_dir: Path, patterns):
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def apply_model_dir_overrides(infer_cfg: dict, root: Path, model_dir: Path):
    # 大将军，这里按目录自动找 pth 和 scaler，方便只改一个模型目录就能跑。
    pth_path = find_one_file(model_dir, ["*.pth", "**/*.pth"])
    scaler_path = find_one_file(model_dir, ["scaler*.pkl", "**/scaler*.pkl", "*.pkl", "**/*.pkl"])
    if pth_path is None:
        raise FileNotFoundError("模型目录里没有找到 .pth: {}".format(model_dir))
    if scaler_path is None:
        raise FileNotFoundError("模型目录里没有找到 scaler .pkl: {}".format(model_dir))

    cfg = json.loads(json.dumps(infer_cfg))
    cfg["model"]["weights_path"] = str(pth_path)
    cfg["normalization"]["scaler_path"] = str(scaler_path)
    return cfg, pth_path, scaler_path


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("{} must be > 0, got {}".format(field_name, parsed))
    return parsed


def build_dataset_with_sources(data_dir: Path, data_cfg: dict, feature_mode: str):
    # 这里和 infer.py 的切窗逻辑保持一致，只额外记录样本来自哪个 CSV。
    base_window_size = require_positive_int(data_cfg["base_window_size"], "data.base_window_size")
    base_step = require_positive_int(data_cfg["base_step"], "data.base_step")
    seq_length = require_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    seq_step = require_positive_int(data_cfg["sequence_step"], "data.sequence_step")

    X_list = []
    y_list = []
    source_list = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        signal = infer.read_signal(csv_file)
        if signal.size < base_window_size:
            continue
        features = []
        for start in range(0, signal.size - base_window_size + 1, base_step):
            window = signal[start : start + base_window_size]
            window = infer.apply_feature_mode(window.astype(np.float32), feature_mode)
            features.append(window)
        if len(features) < seq_length:
            continue
        label = infer.parse_label_from_name(csv_file.name)
        for i in range(0, len(features) - seq_length + 1, seq_step):
            X_list.append(np.stack(features[i : i + seq_length], axis=0))
            y_list.append(label)
            source_list.append(csv_file.name)

    if not X_list:
        return (
            np.empty((0, seq_length, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(source_list, dtype=object),
    )


def build_scaled_dataset(root: Path, infer_cfg: dict, data_dir: Path):
    data_cfg = infer_cfg["data"]
    feature_mode = infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw"))
    X, y, source = build_dataset_with_sources(data_dir, data_cfg, feature_mode)
    if X.shape[0] == 0:
        raise RuntimeError("没有有效样本: {}".format(data_dir))
    scaler_path = resolve_under_root(root, infer_cfg["normalization"]["scaler_path"])
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    return X_scaled, y.astype(np.float32), source, feature_mode


def run_pth_predictions(X_scaled: np.ndarray, infer_cfg: dict, root: Path):
    model_cfg = infer_cfg["model"]
    model_path = resolve_under_root(root, model_cfg["weights_path"])
    state_dict = infer.load_state_dict_compat(model_path, torch.device("cpu"))
    model = infer.build_model_from_config(
        model_cfg=model_cfg,
        input_shape=tuple(X_scaled.shape[1:]),
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_scaled)).cpu().numpy().reshape(-1).astype(np.float32)


def make_prediction_rows(source, y_true, pred):
    rows = []
    for idx in range(len(y_true)):
        rows.append([int(idx), str(source[idx]), float(y_true[idx]), float(pred[idx]), float(abs(pred[idx] - y_true[idx]))])
    return rows


def make_per_csv_rows(source, y_true, pred):
    rows = []
    for name in list(dict.fromkeys(source.tolist())):
        mask = source == name
        diff = pred[mask] - y_true[mask]
        abs_diff = np.abs(diff)
        rows.append(
            [
                name,
                float(y_true[mask][0]),
                int(np.sum(mask)),
                float(np.mean(pred[mask])),
                float(np.mean(abs_diff)),
                float(np.sqrt(np.mean(diff ** 2))),
                float(np.max(abs_diff)),
            ]
        )
    rows.sort(key=lambda row: row[4], reverse=True)
    return rows


def make_per_dryness_rows(y_true, pred):
    rows = []
    for dryness in sorted({float(v) for v in y_true.tolist()}):
        mask = y_true == np.float32(dryness)
        diff = pred[mask] - y_true[mask]
        abs_diff = np.abs(diff)
        rows.append(
            [
                float(dryness),
                int(np.sum(mask)),
                float(np.mean(pred[mask])),
                float(np.mean(abs_diff)),
                float(np.sqrt(np.mean(diff ** 2))),
                float(np.max(abs_diff)),
            ]
        )
    rows.sort(key=lambda row: row[3], reverse=True)
    return rows


def run_pth_compare(infer_config: Path, model_dir: Path, data_dir: Path, output_dir: Path, log):
    root = Path(__file__).resolve().parent
    infer_cfg = load_json(infer_config)
    infer_cfg, pth_path, scaler_path = apply_model_dir_overrides(infer_cfg, root, model_dir)

    tag = model_tag(pth_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions_pth_compare__{}.csv".format(tag)
    per_csv_csv = output_dir / "predictions_pth_per_csv__{}.csv".format(tag)
    per_dryness_csv = output_dir / "predictions_pth_per_dryness__{}.csv".format(tag)
    summary_json = output_dir / "predictions_pth_summary__{}.json".format(tag)

    t0 = time.perf_counter()
    log("读取数据并标准化: {}".format(data_dir))
    X_scaled, y_true, source, feature_mode = build_scaled_dataset(root, infer_cfg, data_dir)
    log("样本数: {}, input_shape: {}".format(X_scaled.shape[0], tuple(X_scaled.shape[1:])))

    t_pth = time.perf_counter()
    pred = run_pth_predictions(X_scaled, infer_cfg, root)
    pth_sec = time.perf_counter() - t_pth

    diff = pred - y_true
    abs_diff = np.abs(diff)
    summary = {
        "total_samples": int(len(y_true)),
        "csv_file_count": int(len(set(source.tolist()))),
        "feature_mode": feature_mode,
        "infer_config": str(infer_config),
        "model_dir": str(model_dir),
        "pth_path": str(pth_path),
        "scaler_path": str(scaler_path),
        "data_dir": str(data_dir),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_error": float(np.max(abs_diff)),
        "pth_infer_time_sec": float(pth_sec),
        "pipeline_total_time_sec": float(time.perf_counter() - t0),
        "predictions_csv": str(predictions_csv),
        "per_csv_csv": str(per_csv_csv),
        "per_dryness_csv": str(per_dryness_csv),
    }

    save_rows_csv(predictions_csv, ["sample_id", "csv_name", "true_label", "prediction", "abs_error"], make_prediction_rows(source, y_true, pred))
    save_rows_csv(per_csv_csv, ["csv_name", "true_label", "sample_count", "pred_mean", "mae", "rmse", "max_abs_error"], make_per_csv_rows(source, y_true, pred))
    save_rows_csv(per_dryness_csv, ["dryness_label", "sample_count", "pred_mean", "mae", "rmse", "max_abs_error"], make_per_dryness_rows(y_true, pred))
    save_json(summary_json, summary)

    log("完成")
    log("MAE: {:.8f}, RMSE: {:.8f}, max_abs_error: {:.8f}".format(summary["mae"], summary["rmse"], summary["max_abs_error"]))
    log("prediction_csv: {}".format(predictions_csv))
    log("per_csv_csv: {}".format(per_csv_csv))
    log("per_dryness_csv: {}".format(per_dryness_csv))
    log("summary_json: {}".format(summary_json))


class App:
    def __init__(self, root_window):
        self.root_window = root_window
        self.root_window.title("PTH 干度预测输出")
        self.project_root = Path(__file__).resolve().parent
        default_cfg = (self.project_root / DEFAULT_INFER_CONFIG).resolve()

        self.infer_config = StringVar(value=str(default_cfg))
        self.model_dir = StringVar(value=str((self.project_root / "model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057").resolve()))
        self.data_dir = StringVar(value=str((self.project_root / "data/880k_data_260414").resolve()))
        self.output_dir = StringVar(value=str((self.project_root / DEFAULT_OUTPUT_DIR).resolve()))

        self._row = 0
        self.add_path_row("infer 配置", self.infer_config, self.pick_file)
        self.add_path_row("pth 模型目录", self.model_dir, self.pick_dir)
        self.add_path_row("测试数据目录", self.data_dir, self.pick_dir)
        self.add_path_row("输出目录", self.output_dir, self.pick_dir)

        self.run_button = Button(root_window, text="运行", command=self.start_run, width=16)
        self.run_button.grid(row=self._row, column=1, pady=8, sticky="w")
        self._row += 1

        self.log_box = Text(root_window, width=110, height=24)
        self.log_box.grid(row=self._row, column=0, columnspan=3, padx=8, pady=8)

    def add_path_row(self, title, variable, picker):
        Label(self.root_window, text=title, width=14, anchor="e").grid(row=self._row, column=0, padx=6, pady=4)
        Entry(self.root_window, textvariable=variable, width=92).grid(row=self._row, column=1, padx=6, pady=4)
        Button(self.root_window, text="选择", command=lambda: picker(variable)).grid(row=self._row, column=2, padx=6, pady=4)
        self._row += 1

    def pick_file(self, variable):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def pick_dir(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def log(self, text):
        def append():
            self.log_box.insert(END, str(text) + "\n")
            self.log_box.see(END)

        self.root_window.after(0, append)

    def start_run(self):
        self.run_button.config(state="disabled")
        self.log_box.delete("1.0", END)

        def worker():
            try:
                run_pth_compare(
                    infer_config=Path(self.infer_config.get()).resolve(),
                    model_dir=Path(self.model_dir.get()).resolve(),
                    data_dir=Path(self.data_dir.get()).resolve(),
                    output_dir=Path(self.output_dir.get()).resolve(),
                    log=self.log,
                )
                self.root_window.after(0, lambda: messagebox.showinfo("完成", "PTH 预测输出完成"))
            except Exception:
                err = traceback.format_exc()
                self.log(err)
                self.root_window.after(0, lambda: messagebox.showerror("失败", err))
            finally:
                self.root_window.after(0, lambda: self.run_button.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root_window = Tk()
    App(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
