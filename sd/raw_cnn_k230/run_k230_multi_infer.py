"""K230 多模型兼容入口。

实际运行框架已经迁入 runtime 包：
- UART 在线：runtime.online
- CSV 离线：runtime.csv

这个文件只保留旧脚本名，避免历史部署或手工调试命令失效。
"""


OVERRIDE_CONFIG_PATH = None


def run_multi_uart_online(cfg, root, uart_sender):
    """中文注释：旧函数名兼容，实际转到统一 UART 在线后端。"""
    from runtime import online as runtime_online

    return runtime_online.run_uart_online(
        cfg=cfg,
        root=root,
        uart_sender=uart_sender,
    )


def run_multi_csv_cached(cfg, root):
    """中文注释：旧函数名兼容，实际转到统一 CSV 后端。"""
    from runtime import csv as runtime_csv

    return runtime_csv.run_csv_cached(
        cfg=cfg,
        root=root,
    )


def main():
    """中文注释：旧脚本入口兼容，实际走唯一 runtime 入口。"""
    import runtime.app as runtime_app

    runtime_app.main(config_path=OVERRIDE_CONFIG_PATH)


if __name__ == "__main__":
    main()
