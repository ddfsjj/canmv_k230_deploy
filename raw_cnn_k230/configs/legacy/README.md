# K230 Legacy Configs

这个目录只保留旧脚本时代的 `k230_config_*` 和 `auto_start_config.json`，用于追溯历史流程或显式兼容调试。

正式板端部署默认只读取：

```text
raw_cnn_k230/configs/runtime.json
```

新增模型、切换单模型/多模型、调整输入通道、输出槽位、异常阈值和 UART 协议时，优先修改 `runtime.json`，不要在旧配置里继续扩展新逻辑。
