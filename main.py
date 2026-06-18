import sys

APP_DIR = "/sdcard/raw_cnn_k230"

# 中文注释：外层 main.py 只负责把项目目录加入模块路径，
# 真正的单模型/多模型选择逻辑都放到项目目录里的 main.py。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import main
