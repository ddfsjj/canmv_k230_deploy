"""
K230 上电自启入口。

这份入口脚本只负责一件事：启动统一 runtime。

设计原则：
1. 板端默认只读 configs/runtime.json。
2. 单模型/多模型都由 runtime.json 的 models[] 决定。
3. 外层保留异常重启循环，避免业务脚本报错后整板停死。
"""

import sys
import time
try:
    import uos as os  # type: ignore
except ImportError:
    import os

APP_DIR = "/sdcard/raw_cnn_k230"
RUNTIME_CONFIG_PATH = "/sdcard/raw_cnn_k230/configs/runtime.json"

# 中文注释：保证后续可以直接导入应用目录下的脚本。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def run_once():
    """中文注释：执行一次统一 runtime 自启。"""
    print("Auto-start mode: unified")
    print("Auto-start config:", RUNTIME_CONFIG_PATH)
    import run_k230_infer as infer_app
    infer_app.OVERRIDE_CONFIG_PATH = RUNTIME_CONFIG_PATH
    infer_app.main()


def print_startup_error(exc):
    print("UART auto-start error:", exc)
    print("APP_DIR:", APP_DIR)
    print("RUNTIME_CONFIG_PATH:", RUNTIME_CONFIG_PATH)
    try:
        print("cwd:", os.getcwd())
    except Exception as cwd_exc:
        print("cwd unavailable:", cwd_exc)


while True:
    try:
        run_once()
    except Exception as exc:
        # 中文注释：启动类程序不能因为一次异常就直接退出，
        # 否则板子上电后会停在错误状态，只能人工重启。
        print_startup_error(exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
