"""
K230 上电自启动入口。

设计思路：
1. 根目录 `main.py` 负责进入应用目录。
2. 真正的业务逻辑统一收敛到 `run_k230_infer.py`。
3. 外层保留一个死循环，业务异常时延时后自动重启，避免板子停死。
"""

import sys
import time

APP_DIR = "/sdcard/raw_cnn_k230"

# 保证后续可以直接导入应用目录下的脚本。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def run_once():
    # 每次进入一次完整的业务主流程。
    # 实际跑哪种模式，由 run_k230_infer.py + k230_config.json 决定。
    import run_k230_infer as infer_app

    infer_app.main()


while True:
    try:
        run_once()
    except Exception as exc:
        # 启动类程序不能因为一次异常就彻底退出，
        # 因此这里捕获异常后稍等再重启。
        print("UART auto-start error:", exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
