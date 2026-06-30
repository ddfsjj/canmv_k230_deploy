# Legacy Inventory

这份清单用于说明哪些文件仍保留在仓库里，但不属于当前正式部署主线。

原则：

```text
正式部署只看 raw_cnn_k230/configs/runtime.json
正式打包只看 scripts/make_deploy_package.py 收集到的文件
历史配置和旧说明只作为参考，不再作为修改入口
```

## 正式主线

板端正式入口：

```text
/sdcard/main.py
  -> /sdcard/raw_cnn_k230/run_k230_infer.py
  -> raw_cnn_k230/runtime/app.py
  -> raw_cnn_k230/configs/runtime.json
```

正式部署包由以下命令生成：

```bash
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

部署包只会自动收集 `runtime.json` 引用到的 KModel 和 scaler。

## 历史板端配置

以下文件保留为历史参考或旧入口兼容，不作为正式修改入口：

```text
raw_cnn_k230/configs/legacy/k230_config_cnn.json
raw_cnn_k230/configs/legacy/k230_config_cnn_lstm.json
raw_cnn_k230/configs/legacy/k230_config_cnn_tcn.json
raw_cnn_k230/configs/legacy/k230_config_cnn_tcn_seg3.json
raw_cnn_k230/configs/legacy/k230_config_multi.json
raw_cnn_k230/configs/legacy/auto_start_config.json
```

如果要换模型、改通道、改单模型/多模型、改输出槽位，应修改：

```text
raw_cnn_k230/configs/runtime.json
```

## 兼容入口

以下文件仍会保留：

```text
raw_cnn_k230/run_k230_infer.py
raw_cnn_k230/run_k230_multi_infer.py
raw_cnn_k230/main.py
raw_cnn_k230/boot.py
```

规则：

```text
run_k230_infer.py        正式薄入口，只负责转到 runtime.app
run_k230_multi_infer.py  兼容入口，旧函数名内部转到 runtime.online/runtime.csv
legacy/run_k230_infer_legacy.py  历史后端函数，仅供旧调试/对比工具兼容使用
main.py                  应用目录内自启入口
boot.py                  应用目录内路径初始化入口
```

新业务逻辑不要继续加到旧入口主体里：

```text
UART 协议类 -> runtime/uart.py
输入读取和日志 -> runtime/inputs.py
窗口维护 -> runtime/windows.py
模型绑定 -> runtime/bindings.py
异常状态 -> runtime/status.py
协议常量和打包 -> runtime/protocol.py
输出协议 -> runtime/output.py
配置转换 -> runtime/config.py
启动编排 -> runtime/app.py
UART 在线主循环 -> runtime/online.py
CSV 离线后端 -> runtime/csv.py
```

## 历史 PC 文档

以下文档可能仍提到旧 `k230_config_*.json` 或旧脚本路径，只作为历史流程参考：

```text
raw_cnn_pc/README.md
raw_cnn_pc/RAW_CNN_PC_K230_完整流程说明.md
raw_cnn_pc/目录结构说明.md
SCRIPT_OVERVIEW.md
```

当前总入口以仓库根目录 `README.md` 和 `docs/` 为准。

## 模型资产

`raw_cnn_k230/model/` 下可能保留多版历史 KModel、ONNX 和 scaler。正式部署包不会整目录打包，只会打包：

```text
runtime.json -> models[].assets.kmodel
runtime.json -> models[].assets.scaler_json
```

因此现场部署前以 `DEPLOY_MANIFEST.json` 为准，不以 `raw_cnn_k230/model/` 目录里存在的全部文件为准。

## 后续清理建议

等板端连续稳定后再做物理删除或移动：

```text
1. 保留兼容入口，继续禁止新增业务逻辑
2. 旧 k230_config_*.json 已移入 raw_cnn_k230/configs/legacy/，后续稳定后再决定是否删除
3. 将旧 PC 流程文档移入 docs/legacy/ 或删除
4. run_k230_multi_infer.py 已压成兼容入口，继续保留一段时间观察现场稳定性
5. run_k230_infer.py 已瘦身为薄入口；历史函数已移入 raw_cnn_k230/legacy/
6. 最后删除不再被 runtime 和部署脚本引用的旧代码
```
