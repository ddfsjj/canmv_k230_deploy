"""
K230 启动早期脚本。

这个文件会在主程序之前执行，作用只有一个：
把 `/sdcard/raw_cnn_k230` 加入模块搜索路径。

这样板端根目录只保留很薄的一层启动入口，
真正的业务脚本都放在 `raw_cnn_k230/` 目录里维护。
"""

import sys

APP_DIR = "/sdcard/raw_cnn_k230"

# 只在路径里还没有该目录时才插入，避免重复追加。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
