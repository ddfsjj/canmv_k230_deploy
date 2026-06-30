"""PC 端 scaler 加载、应用和板端 JSON 导出。"""

from pathlib import Path

import numpy as np

from raw_cnn.config import save_json


def load_scaler(scaler_pkl: Path):
    import joblib

    return joblib.load(scaler_pkl)


def apply_scaler(scaler_pkl: Path, X: np.ndarray):
    scaler = load_scaler(scaler_pkl)
    X_flat = X.reshape(-1, X.shape[-1])
    return scaler.transform(X_flat).reshape(X.shape).astype(np.float32)


def export_scaler_json(scaler_pkl: Path, scaler_json: Path):
    # 中文注释：K230 板端不能读 sklearn pkl，所以导出纯 JSON 参数。
    scaler = load_scaler(scaler_pkl)
    payload = {
        "type": "StandardScaler",
        "n_features_in": int(getattr(scaler, "n_features_in_", 0)),
        "mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
    }
    save_json(scaler_json, payload)
