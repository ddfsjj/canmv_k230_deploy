"""平台适配工具。

这里集中放置时间、路径和轻量文件系统函数，兼容 CanMV/MicroPython 与 PC Python。
"""

import time

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore


def now_us():
    """返回微秒时间戳，用于统计推理耗时。"""
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    return int(time.perf_counter() * 1000000)


def diff_us(t_end, t_start):
    """兼容不同运行时下的时间差计算接口。"""
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(t_end, t_start)
    return t_end - t_start


def sleep_ms(ms):
    """统一毫秒级 sleep。"""
    value = int(ms)
    if value <= 0:
        return
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(value)
    else:
        time.sleep(float(value) / 1000.0)


def drain_uart_rx(uart, empty_rounds=3, sleep_between_ms=10):
    """启动前清空 UART 接收缓冲，连续多次读空后结束。"""
    total_bytes = 0
    empty_hits = 0
    rounds_need = int(empty_rounds)
    if rounds_need <= 0:
        rounds_need = 1
    while empty_hits < rounds_need:
        data = uart.read()
        if data:
            total_bytes += len(data)
            empty_hits = 0
        else:
            empty_hits += 1
    return total_bytes


def file_size_mtime(path):
    """返回文件大小和修改时间，用于缓存键。"""
    try:
        st = os.stat(path)
    except OSError:
        return 0, 0
    try:
        size = int(st[6])
    except Exception:
        size = 0
    try:
        mtime = int(st[8])
    except Exception:
        try:
            mtime = int(st[-2])
        except Exception:
            mtime = 0
    return size, mtime


def norm_path(path):
    """统一路径分隔符。"""
    return str(path).replace("\\", "/")


def join_path(base, rel):
    """轻量路径拼接，适配板端文件系统。"""
    rel = norm_path(rel)
    if rel.startswith("/") or (len(rel) > 2 and rel[1] == ":" and rel[2] == "/"):
        return rel
    base = norm_path(base)
    if base.endswith("/"):
        return base + rel
    return base + "/" + rel


def dirname(path):
    """轻量 dirname 实现。"""
    p = norm_path(path).rstrip("/")
    idx = p.rfind("/")
    if idx < 0:
        return "."
    if idx == 0:
        return "/"
    return p[:idx]


def exists(path):
    """通过 stat 判断路径是否存在。"""
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def ensure_dir(path):
    """逐级创建目录，忽略已存在目录。"""
    p = norm_path(path)
    if p in {"", ".", "/"}:
        return
    abs_path = p.startswith("/")
    cur = "/" if abs_path else ""
    parts = [seg for seg in p.strip("/").split("/") if seg]
    for seg in parts:
        if cur == "/":
            cur = "/" + seg
        elif cur == "":
            cur = seg
        else:
            cur = cur + "/" + seg
        try:
            os.mkdir(cur)
        except OSError:
            pass


def list_csv_files(data_dir):
    """列出 CSV 文件；配置指向单个 CSV 时也支持。"""
    if str(data_dir).lower().endswith(".csv"):
        if exists(data_dir):
            return [data_dir]
        return []
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    files = []
    for name in names:
        if str(name).lower().endswith(".csv"):
            files.append(join_path(data_dir, name))
    files.sort()
    return files


def file_stem(path):
    """取文件主名，不带扩展名。"""
    name = norm_path(path).split("/")[-1]
    dot = name.rfind(".")
    if dot > 0:
        return name[:dot]
    return name
