# K230 Configs

正式板端运行配置只有：

```text
runtime.json
```

单模型、多模型、模型文件、scaler、输入通道、输出槽位、异常阈值和 UART 协议都从 `runtime.json` 修改。

示例配置：

```text
runtime.single.example.json
runtime.multi.example.json
```

这些只用于复制参考，不会被默认部署脚本打包。

历史兼容配置放在：

```text
legacy/
  auto_start_config.json
  k230_config_cnn.json
  k230_config_cnn_lstm.json
  k230_config_cnn_tcn.json
  k230_config_cnn_tcn_seg3.json
  k230_config_multi.json
```

这些文件只作为旧流程参考或兼容入口参考。正式部署不读取它们，也不会由 `scripts/make_deploy_package.py` 默认打包。
