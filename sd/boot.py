"""SD 卡根目录 boot.py：只负责把应用目录加入模块搜索路径。"""

import sys

APP_DIR = "/sdcard/raw_cnn_k230"

# 中文注释：根目录启动器保持很薄，业务代码统一放在 raw_cnn_k230 目录。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
