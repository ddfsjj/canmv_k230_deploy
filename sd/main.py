"""SD 卡根目录 main.py：上电后启动 raw_cnn_k230 统一 runtime。"""

import sys
import time
try:
    import uos as os  # type: ignore
except ImportError:
    import os

APP_DIR = "/sdcard/raw_cnn_k230"
RUNTIME_CONFIG_PATH = "/sdcard/raw_cnn_k230/configs/runtime.json"

# 中文注释：先把应用目录放到搜索路径最前面，再导入业务入口。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def run_once():
    print("SD root launcher: raw_cnn_k230 unified runtime")
    print("runtime_config:", RUNTIME_CONFIG_PATH)
    import run_k230_infer as infer_app
    infer_app.OVERRIDE_CONFIG_PATH = RUNTIME_CONFIG_PATH
    infer_app.main()


def print_startup_error(exc):
    print("SD root auto-start error:", exc)
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
        # 中文注释：上电自启入口不直接退出，避免一次异常后板端停在空状态。
        print_startup_error(exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
