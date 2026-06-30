"""数值数组工具。

这里集中处理 numpy/ulab 的浮点数组兼容、环形缓冲展开和轻量统计。
"""

try:
    import ulab.numpy as np  # type: ignore
except ImportError:
    import numpy as np  # type: ignore

NP_FLOAT = getattr(np, "float32", None)
if NP_FLOAT is None:
    NP_FLOAT = float


def as_float_array(values):
    """尽量把输入转换为浮点数组。"""
    try:
        return np.asarray(values, dtype=NP_FLOAT)
    except TypeError:
        return np.asarray(values)


def astype_float_array(arr):
    """在 numpy/ulab 间兼容 astype(dtype)。"""
    if not hasattr(arr, "astype"):
        return arr
    try:
        return arr.astype(NP_FLOAT)
    except TypeError:
        return arr


def empty_float(shape):
    """申请浮点数组。"""
    try:
        return np.empty(shape, dtype=NP_FLOAT)
    except TypeError:
        return np.empty(shape)


def expand_ring_window(ring_row, write_idx, out_window):
    """将一路环形缓冲展开为最旧到最新的连续窗口。"""
    n = int(len(ring_row))
    idx = int(write_idx) % n
    if idx == 0:
        out_window[:] = ring_row
        return out_window
    right = n - idx
    out_window[:right] = ring_row[idx:]
    out_window[right:] = ring_row[:idx]
    return out_window


def expand_sequence_ring(seq_ring, write_idx, out_seq):
    """将序列环形缓冲展开为最旧到最新的连续序列。"""
    n = int(len(seq_ring))
    idx = int(write_idx) % n
    if idx == 0:
        out_seq[:] = seq_ring
        return out_seq
    right = n - idx
    out_seq[:right] = seq_ring[idx:]
    out_seq[right:] = seq_ring[:idx]
    return out_seq


def mean_1d(values):
    """手写一维均值，兼容板端运行环境。"""
    count = int(len(values))
    if count <= 0:
        return 0.0
    total = 0.0
    for i in range(count):
        total += float(values[i])
    return total / float(count)
