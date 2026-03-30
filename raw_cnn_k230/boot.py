"""
K230 启动早期脚本。

这个文件会在主程序之前执行，用来把应用目录加入模块搜索路径。
这样根目录 `/sdcard/main.py` 就可以直接 `import run_k230_infer`，
而不需要把所有业务脚本都平铺放在根目录。
"""

import sys

APP_DIR = "/sdcard/raw_cnn_k230"

# 只在路径中不存在时插入，避免重复追加。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
