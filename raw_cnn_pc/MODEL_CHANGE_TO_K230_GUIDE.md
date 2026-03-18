# Raw CNN 模型替换与 K230 部署操作说明

本文档说明两件事：
- 换了 CNN 模型后需要改哪些文件。
- 改完后如何导出、拷贝、在 K230 上运行。

## 1. 先判断改动类型

### A) 只改路径（最小改动）

满足以下条件时，通常只需要改路径：
- 还是同一套网络定义（`CNNAll`）。
- `conv_filters`、`kernel_size`、`pool_size` 没变。
- 数据切窗参数没变：`base_window_size`、`base_step`、`sequence_length`、`sequence_step`。
- 归一化方式没变（仍是 `StandardScaler`）。

你只需要把新的 `*.pth` 和 `*.pkl` 放进 `raw_cnn_pc/model/`，并更新配置里的路径。

### B) 改配置（不改代码）

仍然是 `CNNAll`，但你改了下面任意参数：
- `conv_filters` / `kernel_size` / `pool_size`
- `base_window_size` / `base_step` / `sequence_length` / `sequence_step`
- 量化样本数、量化参数

这种情况要同步更新多个 `json`，否则会出现维度不一致或结果异常。

### C) 改代码（模型结构变更）

如果你不再使用当前 `CNNAll` 结构，或前处理流程不是当前 raw + scaler 流程，需要改代码：
- `raw_cnn_pc/infer.py`
- `raw_cnn_pc/build_kmodel.py`
- 必要时 `raw_cnn_k230/run_k230_infer.py`

典型信号：`load_state_dict(strict=True)` 报 key 或 shape 不匹配。

## 2. 必改文件清单

## 2.1 PC 侧推理配置

文件：`raw_cnn_pc/infer_config.json`

关注字段：
- `model.weights_path`
- `normalization.scaler_path`
- `model.conv_filters`
- `model.kernel_size`
- `model.pool_size`
- `data.*`（切窗/序列参数）

## 2.2 导出到 K230 配置

文件：`raw_cnn_pc/k230_export_config.json`

关注字段：
- `paths.weights_pth`
- `paths.scaler_pkl`
- `paths.onnx`
- `paths.kmodel`
- `paths.scaler_json`
- `data.*`
- `model.*`
- `quantization.*`

## 2.3 K230 运行时配置

文件：`raw_cnn_k230/k230_config.json`

关注字段：
- `paths.kmodel`
- `paths.scaler_json`
- `paths.test_data_dir`
- `data.*`（必须与导出时保持一致）
- `runtime.max_samples`（快速测试可设 10）
- `uart.enabled`（不接 MCU 可设为 `false`）

## 3. 修改后的完整操作流程

以下命令默认在仓库根目录 `canmv_k230_deploy` 执行。

## 3.1 安装依赖（首次或环境变化后）

```bash
cd raw_cnn_pc
pip install -r requirements.txt
pip install -r requirements_k230_host.txt
```

如遇 `Failed to get hostfxr path`，先安装 .NET Runtime 7：

```bash
winget install --id Microsoft.DotNet.Runtime.7 -e --silent --accept-package-agreements --accept-source-agreements
```

## 3.2 PC 快速自检（强烈建议）

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --max_samples 10 --output predictions_pc_quick.csv
```

看到正常输出并生成 `predictions_pc_quick.csv` 再继续。

## 3.3 导出 ONNX / scaler / calibration / kmodel

完整导出并编译：

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json
```

仅导出不编译（排查时可用）：

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json --skip_compile
```

导出产物会写到 `raw_cnn_k230/model/`（由 `k230_export_config.json` 的 `paths.*` 决定）。

## 3.4 拷贝到 K230

推荐直接拷贝整个目录：
- `raw_cnn_k230/` -> `/sdcard/raw_cnn_k230/`

至少要拷贝这些内容：
- `run_k230_infer.py`
- `k230_config.json`
- `model/*.kmodel`
- `model/*.json`
- `test_data/*.csv`（如果板端需要本地测试）

## 3.5 板端运行

在 K230 终端执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

运行完成后关注：
- 控制台是否打印 `samples`、`MAE`、`RMSE`
- 是否生成 `predictions_k230.csv`
- 开启 UART 时是否有 `uart_sent_frames`

## 4. 每次换模型的最短检查清单

- 新的 `pth` 和 `pkl` 已放到 `raw_cnn_pc/model/`。
- `infer_config.json` 与 `k230_export_config.json` 中模型和数据参数一致。
- `raw_cnn_k230/k230_config.json` 的 `data.*` 与导出参数一致。
- `python infer.py --max_samples 10` 已通过。
- `python build_kmodel.py` 已生成新的 `.kmodel` 和 `scaler_run*.json`。
- 已把更新后的 `raw_cnn_k230/` 整体拷到板子。

## 5. 常见问题

- 权重加载失败（shape/key 不匹配）：
  模型结构变了，不是“只改路径”，需同步改 `model.*` 参数，必要时改代码。
- 板端报输入维度错误：
  `raw_cnn_k230/k230_config.json` 的 `data.*` 与导出时不一致。
- 推理很慢或先只想看流程：
  把 `runtime.max_samples` 先设为 `10`。
- MCU 没接收数据：
  检查 `uart.enabled`、串口号、引脚、波特率和帧格式（`AA + 12xint32 + FF`）。
