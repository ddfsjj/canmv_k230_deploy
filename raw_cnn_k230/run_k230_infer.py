"""K230 统一 runtime 薄入口。

业务实现已经迁入 runtime 包；本文件只保留历史脚本名，方便 SD 卡启动器、
CanMV IDE 和旧命令继续运行 `run_k230_infer.py`。
"""

OVERRIDE_CONFIG_PATH = None


def main():
    """中文注释：旧脚本名兼容入口，实际执行 runtime.app。"""
    from runtime import app as runtime_app

    runtime_app.main(config_path=OVERRIDE_CONFIG_PATH)


if __name__ == "__main__":
    main()
