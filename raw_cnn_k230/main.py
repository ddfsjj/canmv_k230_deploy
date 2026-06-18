"""
K230 上电自启入口。

这份入口脚本只负责一件事：根据配置决定本次自启跑单模型还是多模型。

设计原则：
1. 默认行为保持兼容，未配置时仍走单模型。
2. 切换模式尽量只改配置文件，不要求手改 main.py。
3. 外层保留异常重启循环，避免业务脚本报错后整板停死。
"""

import json
import sys
import time

APP_DIR = "/sdcard/raw_cnn_k230"
AUTO_START_CONFIG_PATH = APP_DIR + "/configs/auto_start_config.json"

# 中文注释：保证后续可以直接导入应用目录下的脚本。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def load_auto_start_config():
    """
    中文注释：读取自启配置。

    配置缺失时自动回退到单模型默认值，这样即使用户还没创建配置文件，
    板子也能沿用原先的单模型方式启动，不会因为少一份 json 直接起不来。
    """
    default_cfg = {
        "entry": "single",
        "single_config": "configs/k230_config_cnn_tcn.json",
        "multi_config": "configs/k230_config_multi.json",
    }
    try:
        with open(AUTO_START_CONFIG_PATH, "r") as f:
            loaded = json.load(f)
    except Exception:
        return default_cfg

    if not isinstance(loaded, dict):
        return default_cfg

    merged = dict(default_cfg)
    merged.update(loaded)
    return merged


def normalize_entry_mode(raw_value):
    """
    中文注释：统一自启模式写法。

    这里兼容几种常见写法，避免配置里写成 `multi-model`、`multi_models`
    之类的形式时因为名字不完全一致导致走错分支。
    """
    text = str(raw_value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"multi", "multiple", "multi_model", "multi_models"}:
        return "multi"
    return "single"


def run_once():
    """
    中文注释：执行一次自启。

    选择规则：
    1. `entry = single` 时调用单模型入口 `run_k230_infer.py`
    2. `entry = multi`  时调用多模型入口 `run_k230_multi_infer.py`
    """
    auto_cfg = load_auto_start_config()
    entry = normalize_entry_mode(auto_cfg.get("entry", "single"))

    if entry == "multi":
        config_path = str(auto_cfg.get("multi_config", "configs/k230_config_multi.json"))
        print("Auto-start mode: multi")
        print("Auto-start config:", config_path)
        import run_k230_multi_infer as infer_app
        infer_app.OVERRIDE_CONFIG_PATH = config_path
        infer_app.main()
        return

    config_path = str(auto_cfg.get("single_config", "configs/k230_config_cnn_tcn.json"))
    print("Auto-start mode: single")
    print("Auto-start config:", config_path)
    import run_k230_infer as infer_app
    infer_app.OVERRIDE_CONFIG_PATH = config_path
    infer_app.main()


while True:
    try:
        run_once()
    except Exception as exc:
        # 中文注释：启动类程序不能因为一次异常就直接退出，
        # 否则板子上电后会停在错误状态，只能人工重启。
        print("UART auto-start error:", exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
