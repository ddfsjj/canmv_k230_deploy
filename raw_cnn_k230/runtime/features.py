"""特征预处理和 scaler 参数工具。"""

from runtime import config as runtime_config
from runtime import numeric


def load_scaler_params(scaler_json_path):
    """读取训练阶段导出的 mean / scale，并保护极小 scale。"""
    scaler = runtime_config.load_json(scaler_json_path)
    mean = numeric.as_float_array(scaler["mean"])
    scale = numeric.as_float_array(scaler["scale"])
    eps = 1e-12
    for i in range(len(scale)):
        if abs(float(scale[i])) < eps:
            scale[i] = 1.0
    return mean, scale


def scale_features(X, scaler_json_path):
    """对离线样本执行与在线推理一致的标准化。"""
    mean, scale = load_scaler_params(scaler_json_path)
    X_flat = numeric.astype_float_array(X.reshape((X.shape[0] * X.shape[1], X.shape[-1])))
    X_scaled = (X_flat - mean) / scale
    return numeric.astype_float_array(X_scaled.reshape(X.shape))


def apply_feature_mode_1d(src_window, feature_mode, out_window):
    """按配置把原始窗口转换为模型输入窗口。"""
    mode = runtime_config.normalize_feature_mode(feature_mode)
    if mode == "window_demean":
        mean_value = numeric.mean_1d(src_window)
        out_window[:] = src_window - mean_value
        return out_window
    if mode == "window_rel_demean":
        # 先去窗口均值，再除以均值绝对值，避免不同基线频率直接主导幅度。
        mean_value = numeric.mean_1d(src_window)
        denom = abs(mean_value)
        if denom < 1e-6:
            denom = 1e-6
        out_window[:] = (src_window - mean_value) / denom
        return out_window
    out_window[:] = src_window
    return out_window
