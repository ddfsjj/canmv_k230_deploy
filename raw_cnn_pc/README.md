# Raw CNN PC / K230 使用说明

> 说明：`raw_cnn_pc/` 根目录下旧的 `infer_config.json`、`infer_config_cnn_tcn.json`、`k230_export_config.json`、`k230_export_config_cnn_tcn.json` 已删除。
> 现在统一使用：
> `raw_cnn_pc/configs/infer/*.json`
> `raw_cnn_pc/configs/export/*.json`
> `raw_cnn_k230/configs/*.json`

这套目录分成两部分：

- `raw_cnn_pc/`：在电脑上做 `.pth` 推理、导出 `onnx/kmodel`、对比量化方案。
- `raw_cnn_k230/`：拷到 K230 板子上做离线 CSV 对比，或者做串口在线推理。

目录整理后的归档位置见 `目录结构说明.md`。
历史版本配置现在统一放在 `configs/`，根目录只保留默认入口配置。

如果大将军只想先跑通一遍，直接按下面 4 步做。

## 1. 安装依赖

先装 PC 推理依赖：

```bash
cd raw_cnn_pc
pip install -r requirements.txt
```

如果还要导出 `kmodel`，再装导出依赖：

```bash
cd raw_cnn_pc
pip install -r requirements_k230_host.txt
```

如果 `nncase` 提示缺 `.NET Runtime`，再执行：

```bash
winget install --id Microsoft.DotNet.Runtime.7 -e --silent --accept-package-agreements --accept-source-agreements
```

## 2. 先在 PC 端验证 `.pth`

先确认这两个路径是对的：

- `configs/infer/infer_config_cnn_tcn.json` 里的 `model.weights_path`
- `configs/infer/infer_config_cnn_tcn.json` 里的 `normalization.scaler_path`

然后执行：

```bash
cd raw_cnn_pc
python infer.py --config configs/infer/infer_config_cnn_tcn.json
```

常用附加参数：

```bash
python infer.py --config configs/infer/infer_config_cnn_tcn.json --max_samples 10 --output predictions_pc_quick.csv
```

跑完重点看：

- 是否正常加载模型和 `scaler`
- `samples` 是否符合预期
- `MAE` / `RMSE` 是否正常
- 是否生成预测结果 CSV

## 3. 导出给 K230 用的模型

最常用命令：

```bash
cd raw_cnn_pc
python build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json
```

如果这次只想先验证导出前半段，不想真的编译 `kmodel`：

```bash
python build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json --skip_compile
```

导出完成后，通常会在 `../raw_cnn_k230/model/` 看到：

- `.onnx`
- `.kmodel`
- `scaler json`
- `calibration_input.npy`

## 4. 在 K230 端运行

把整个 `raw_cnn_k230/` 目录拷到板子，例如：

- 本地目录：`raw_cnn_k230/`
- 板端目录：`/sdcard/raw_cnn_k230/`

### 板端离线 CSV 对比

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

### 板端串口在线推理

先把 `k230_config.json` 里的 `runtime.mode` 改成 `uart_online`，再执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

## 三个最重要的配置文件

### 1. `configs/infer/infer_config_cnn_tcn.json`

控制 PC 端 `.pth` 推理：

- 用哪个模型
- 用哪个 `scaler.pkl`
- 测试数据目录在哪
- 默认跑多少样本

最常改的字段：

```json
{
  "data": {
    "test_data_dir": "test_data"
  },
  "model": {
    "weights_path": "model/xxx.pth"
  },
  "normalization": {
    "scaler_path": "model/xxx.pkl"
  },
  "runtime": {
    "max_samples": 10
  }
}
```

### 2. `configs/export/k230_export_config_cnn_tcn.json`

控制从 `.pth` 导出到 `kmodel`：

- 导出用的模型和 `scaler`
- 输出文件名写到哪里
- 量化校准数据从哪里来
- 校准样本抽多少
- 量化类型怎么选

最常改的字段：

```json
{
  "paths": {
    "weights_pth": "model/xxx.pth",
    "scaler_pkl": "model/xxx.pkl",
    "calibration_data_dir": "../generated_dry_temp_csv",
    "onnx": "../raw_cnn_k230/model/xxx.onnx",
    "kmodel": "../raw_cnn_k230/model/xxx.kmodel",
    "scaler_json": "../raw_cnn_k230/model/xxx.json"
  },
  "quantization": {
    "samples_count": 256,
    "sampling_strategy": "first",
    "quant_type": "uint8",
    "weight_quant_type": "int16",
    "calibrate_method": "Kld"
  }
}
```

### 3. `../raw_cnn_k230/k230_config.json`

控制 K230 端运行：

- 板端加载哪个 `kmodel`
- 板端加载哪个 `scaler json`
- 离线 CSV 对比跑多少样本
- 串口在线推理怎么收发
- 现在到底跑哪种模式

最常改的字段：

```json
{
  "paths": {
    "kmodel": "model/xxx.kmodel",
    "scaler_json": "model/xxx.json"
  },
  "runtime": {
    "mode": "uart_online",
    "csv_cached": {
      "compare_max_samples": 10
    }
  }
}
```

## 当前支持的主要功能

### PC 端

- `infer.py`：跑 `.pth` 推理并输出预测 CSV。
- `build_kmodel.py`：导出 `onnx`、`scaler json`、量化校准样本，并编译 `kmodel`。
- `compare_pth_onnx_kmodel_gui.py`：图形界面对比 `.pth / .onnx / .kmodel`。
- `run_pth_csv_compare_gui.py`：图形界面导出 `.pth` 预测结果、逐干度误差和逐文件汇总。
- `evaluate_quantization_schemes.py`：批量评估不同量化方案并生成 Markdown 报告。
- `build_to_k230.bat`：Windows 下一键调用 `build_kmodel.py`。

### K230 端

- `run_k230_infer.py`：统一入口，支持离线和串口在线多种模式。
- `run_k230_csv_compare.py`：固定从第 0 条样本开始，生成板端离线对比 CSV。
- `uart_continuous_send_test.py`：独立串口发包测试，不依赖模型。
- `boot.py` / `main.py`：板子开机自启动入口。

## 最常见的 4 个需求

### 1. 只想换新的 `.pth/.pkl`

通常只要改这几处：

- `configs/infer/infer_config_cnn_tcn.json` 的 `weights_path`、`scaler_path`
- `configs/export/k230_export_config_cnn_tcn.json` 的 `weights_pth`、`scaler_pkl`
- 如果导出的板端文件名也变了，再改 `raw_cnn_k230/k230_config.json` 的 `paths.kmodel`、`paths.scaler_json`

### 2. 只想改 PC 端默认跑多少条

改：

```json
"runtime": {
  "max_samples": 10
}
```

说明：

- 填正整数：只跑前 N 条
- 填 `null`：全量跑

### 3. 只想改板端离线对比默认跑多少条

改：

```json
"runtime": {
  "csv_cached": {
    "compare_max_samples": 10
  }
}
```

### 4. 只想改量化校准样本来源或数量

改：

```json
"paths": {
  "calibration_data_dir": "../generated_dry_temp_csv"
},
"quantization": {
  "samples_count": 256
}
```

## 推荐命令清单

### 快速验证 `.pth`

```bash
python infer.py --config configs/infer/infer_config_cnn_tcn.json --max_samples 10 --output predictions_pc_quick.csv
```

### 正式导出 `kmodel`

```bash
python build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json
```

### 比较 `.pth` 和 `kmodel`

```bash
python compare_pth_onnx_kmodel_gui.py
```

### 批量比较多种量化方案

```bash
python evaluate_quantization_schemes.py --scheme u8_i16_kld_256
python evaluate_quantization_schemes.py --report_only
```

## 常见问题

### `load_state_dict` 报错

通常说明模型结构和权重不匹配。除了路径，`conv_filters`、`kernel_size`、`pool_size`、`sequence_length` 这些也要对上。

### `No valid samples`

通常说明：

- 数据目录错了
- CSV 太短，切不出窗口
- `base_window_size` / `sequence_length` 设得太大

### `onnx package` 缺失

安装：

```bash
pip install -r requirements_k230_host.txt
```

### `nncase is not installed`

通常是当前 Python 环境不对，或者 `.NET Runtime` 没装好。

### 板端能跑，但结果明显不对

优先检查这几项三边是否一致：

- `data.*`
- `preprocessing.feature_mode`
- `scaler`
- 当前加载的 `kmodel/json`

## 更详细的步骤说明

请看：

- `RAW_CNN_PC_K230_完整流程说明.md`
