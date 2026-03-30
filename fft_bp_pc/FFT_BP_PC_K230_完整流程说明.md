# FFT BP（PC + K230）完整流程说明

本文档基于两目录拆分结构：
- `fft_bp_pc/`：PC 侧推理与导出
- `fft_bp_k230/`：K230 板端运行

## 1. 目录分工

`fft_bp_pc/` 负责：
- `infer.py`：在 PC 上验证 `.pth` 推理
- `build_kmodel.py`：导出 `onnx/scaler json/calibration` 并编译 `kmodel`
- 导出结果写入 `../fft_bp_k230/`

`fft_bp_k230/` 负责：
- `run_k230_infer.py`：板端加载 `kmodel` 推理
- `k230_config.json`：板端路径、数据与 FFT 参数

## 2. 核心配置文件

`fft_bp_pc/infer_config.json`（PC 推理）：
- `model.weights_path` / `normalization.scaler_path`
- `data.*`
- `preprocessing.fft_config.*`
- `model.hidden_units/activation/output_activation`

`fft_bp_pc/k230_export_config.json`（导出配置）：
- `paths.weights_pth/scaler_pkl`：导出输入
- `paths.onnx/kmodel/scaler_json/calibration_npy`：导出输出（指向 `../fft_bp_k230`）
- `data.*`、`preprocessing.fft_config.*`、`model.*`
- `quantization.*`

`fft_bp_k230/k230_config.json`（板端运行）：
- `paths.kmodel/scaler_json/test_data_dir/predictions_csv`
- `data.*`、`preprocessing.fft_config.*`

## 3. 一次完整流程

以下命令默认在仓库根目录执行。

### 3.1 安装依赖

```bash
cd fft_bp_pc
pip install -r requirements.txt
pip install -r requirements_k230_host.txt
```

如出现 `Failed to get hostfxr path`：

```bash
winget install --id Microsoft.DotNet.Runtime.7 -e --silent --accept-package-agreements --accept-source-agreements
```

### 3.2 PC 侧快速验证

```bash
cd fft_bp_pc
python infer.py --config infer_config.json --output predictions.csv
```

验证点：
- 无 `load_state_dict` 报错
- 生成 `predictions.csv`
- 日志输出 `samples/input_shape/MAE/RMSE`

### 3.3 导出并编译 K230 产物

```bash
cd fft_bp_pc
python build_kmodel.py --config k230_export_config.json
```

仅导出不编译：

```bash
python build_kmodel.py --config k230_export_config.json --skip_compile
```

临时调整校准样本数：

```bash
python build_kmodel.py --config k230_export_config.json --max_calib_samples 128
```

也可用：

```bash
build_to_k230.bat
```

产物输出到：
- `fft_bp_k230/model/model_run000.onnx`
- `fft_bp_k230/model/model_run000.kmodel`
- `fft_bp_k230/model/scaler_run000.json`
- `fft_bp_k230/model/calibration_input.npy`

### 3.4 拷贝并在板端运行

拷贝：
- `fft_bp_k230/` -> `/sdcard/fft_bp_k230/`

运行：

```python
cd /sdcard/fft_bp_k230
python run_k230_infer.py
```

输出：
- `predictions_k230.csv`
- `samples/input_shape/MAE/RMSE`

## 4. 更换新 pth 模型操作

## 4.1 结构不变（最常见）

条件：
- `hidden_units/activation/output_activation` 不变
- `data.*`、`fft_config.*` 不变

步骤：
1. 新 `*.pth`、`*.pkl` 放入 `fft_bp_pc/model/`
2. 修改 `fft_bp_pc/infer_config.json`：
   - `model.weights_path`
   - `normalization.scaler_path`
3. 修改 `fft_bp_pc/k230_export_config.json`：
   - `paths.weights_pth`
   - `paths.scaler_pkl`
4. 先跑 `python infer.py ...` 验证
5. 再跑 `python build_kmodel.py ...` 导出
6. 拷贝 `fft_bp_k230/` 到板端复测

## 4.2 参数变化（必须同步）

若变更以下任意字段，`infer_config.json` 与 `k230_export_config.json` 必须同步：
- `data.base_window_size/base_step/sequence_length/sequence_step`
- `preprocessing.fft_config.fs/f_min/f_max/window/nfft`
- `model.hidden_units/activation/output_activation`

另外板端 `fft_bp_k230/k230_config.json` 的 `data.*` 与 `fft_config.*` 也必须一致。

## 4.3 模型结构变化（不再兼容 BPNet）

仅改配置不够，需要修改代码：
- `fft_bp_pc/infer.py`
- `fft_bp_pc/build_kmodel.py`
- 必要时 `fft_bp_k230/run_k230_infer.py`

## 5. 常见问题

`ONNX export requires onnx package`：
- 执行 `pip install -r requirements_k230_host.txt`

`nncase is not installed`：
- 检查 `nncase` 是否安装在当前 Python 环境

`No valid samples in test data`：
- `test_data_dir` 路径不对，或样本不满足切窗参数

`load_state_dict` 报 shape/key 不匹配：
- 模型结构参数与权重不一致

板端结果异常：
- 板端 `k230_config.json` 的 `data.*` / `fft_config.*` 与导出配置不一致
