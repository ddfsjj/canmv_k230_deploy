# Project Boundary

这份文档用于区分当前项目里的正式运行入口、兼容入口和历史文件。后续新增功能时，优先改正式区域，不再向旧脚本里扩散逻辑。

## 正式板端入口

板端正式部署结构：

```text
/sdcard/
  boot.py
  main.py
  raw_cnn_k230/
    configs/runtime.json
    runtime/
    run_k230_infer.py
    run_k230_multi_infer.py
    model/...
```

正式入口链路：

```text
/sdcard/main.py
  -> /sdcard/raw_cnn_k230/run_k230_infer.py
  -> raw_cnn_k230/runtime/app.py
  -> configs/runtime.json
  -> runtime inputs/windows/bindings/status/output
```

正式运行期只认：

```text
raw_cnn_k230/configs/runtime.json
raw_cnn_k230/runtime/
raw_cnn_k230/model/
```

## 兼容入口

这些文件短期保留，用于兼容旧调用方式：

```text
raw_cnn_k230/run_k230_infer.py
raw_cnn_k230/run_k230_multi_infer.py
raw_cnn_k230/main.py
raw_cnn_k230/boot.py
```

当前规则：

```text
run_k230_infer.py        作为统一 runtime 薄入口，只负责转到 runtime.app
run_k230_multi_infer.py  作为兼容入口，旧函数名内部转到 runtime.online/runtime.csv
legacy/run_k230_infer_legacy.py  保留旧单模型/对比工具所需历史函数，不进入部署包
main.py                  应用目录内自启入口，主要用于手动运行或非根目录部署
boot.py                  应用目录内路径初始化入口
```

新功能原则：

```text
UART 协议类改 runtime/uart.py
输入读取和日志改 runtime/inputs.py
窗口维护改 runtime/windows.py
模型绑定改 runtime/bindings.py
异常状态改 runtime/status.py
协议常量和打包改 runtime/protocol.py
输出协议改 runtime/output.py
运行配置改 runtime/config.py
启动编排改 runtime/app.py
UART 在线主循环改 runtime/online.py
CSV 离线后端改 runtime/csv.py
```

不要再把新业务逻辑加到旧入口主体里。

## 历史配置

这些配置是旧脚本时代留下的参考文件：

```text
raw_cnn_k230/configs/legacy/k230_config_cnn.json
raw_cnn_k230/configs/legacy/k230_config_cnn_lstm.json
raw_cnn_k230/configs/legacy/k230_config_cnn_tcn.json
raw_cnn_k230/configs/legacy/k230_config_cnn_tcn_seg3.json
raw_cnn_k230/configs/legacy/k230_config_multi.json
raw_cnn_k230/configs/legacy/auto_start_config.json
```

正式部署不读取这些文件。换模型、改单模型/多模型、改通道和改输出槽位，都应该改：

```text
raw_cnn_k230/configs/runtime.json
```

## 示例配置

示例配置只用于复制参考和 PC 校验：

```text
raw_cnn_k230/configs/runtime.single.example.json
raw_cnn_k230/configs/runtime.multi.example.json
```

默认部署脚本不会打包示例配置。

## PC 端边界

PC 端公共包已经承载配置、数据、scaler 和模型定义：

```text
raw_cnn_pc/raw_cnn/config.py
raw_cnn_pc/raw_cnn/data.py
raw_cnn_pc/raw_cnn/scaler.py
raw_cnn_pc/raw_cnn/models.py
```

当前规则：

```text
配置/数据/scaler/model 优先走 raw_cnn_pc/raw_cnn/
模型结构统一维护在 raw_cnn_pc/raw_cnn/models.py
infer.py 只保留 PC 推理流程
build_kmodel.py 只保留 ONNX/KModel 导出流程
```

模型定义差异审计见 [pc_model_definition_audit.md](pc_model_definition_audit.md)。

## 清理顺序

建议等板端连续稳定后再做清理：

```text
1. 保留旧入口，但停止新增逻辑
2. 把旧配置标记为 legacy reference
3. PC 重复模型定义已迁移到公共包
4. run_k230_multi_infer.py 已压成兼容入口
5. run_k230_infer.py 已瘦身为薄入口；历史函数已移入 raw_cnn_k230/legacy/
6. 最后删除或归档旧配置和旧路径
```

历史文件清单见 [legacy_inventory.md](legacy_inventory.md)。
