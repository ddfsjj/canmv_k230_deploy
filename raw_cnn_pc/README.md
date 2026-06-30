# Raw CNN PC

`raw_cnn_pc/` 是 PC 端推理、导出和量化对比目录。当前 PC 端已经抽出公共层，模型结构不再分别维护在
`infer.py` 和 `build_kmodel.py` 里。

## 当前结构

```text
raw_cnn_pc/
  raw_cnn/
    config.py      配置读取和数值校验
    data.py        CSV 读取、标签解析、切窗和特征处理
    scaler.py      scaler 加载、应用和导出 scaler json
    models.py      唯一模型定义层
  configs/
    infer/         PC .pth 推理配置
    export/        ONNX/KModel 导出配置
  infer.py         PC .pth 推理入口
  build_kmodel.py  ONNX/KModel 导出入口
```

当前模型定义统一维护在：

```text
raw_cnn_pc/raw_cnn/models.py
```

已经收口的模型：

```text
CNNAll
CNNLSTM
CNNTCN
CNNTCNSeg3SoftStatsMoE
```

`build_kmodel.py` 仍保留 `Seg3ExportWrapper`，它是导出双输入 ONNX/KModel 的包装器，不是模型结构定义。

## PC 推理

常用命令：

```bash
cd raw_cnn_pc
python infer.py --config configs/infer/infer_config_cnn_tcn.json --max_samples 10 --output predictions_pc_quick.csv
```

常改字段：

```text
configs/infer/*.json
  data.test_data_dir
  model.type
  model.weights_path
  normalization.scaler_path
  runtime.max_samples
```

## 导出 KModel

常用命令：

```bash
cd raw_cnn_pc
python build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json
```

只导出 ONNX/scaler/calibration，不编译 KModel：

```bash
python build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json --skip_compile
```

常改字段：

```text
configs/export/*.json
  paths.weights_pth
  paths.scaler_pkl
  paths.calibration_data_dir
  paths.onnx
  paths.kmodel
  paths.scaler_json
  quantization.samples_count
  quantization.sampling_strategy
  quantization.quant_type
  quantization.weight_quant_type
  quantization.calibrate_method
```

导出的板端资产通常写入：

```text
../raw_cnn_k230/model/...
```

## 更新板端运行配置

导出新 `.kmodel` 和 scaler `.json` 后，正式板端运行配置只改：

```text
raw_cnn_k230/configs/runtime.json
```

常改字段：

```text
models[].assets.kmodel
models[].assets.scaler_json
models[].input_channels
models[].output.slots
models[].window
```

不要再改旧的 `k230_config_*.json` 作为正式部署入口。旧配置只作为 legacy reference。

## 导出后验证

回到仓库根目录运行：

```bash
python scripts/validate_runtime_config.py
python scripts/run_runtime_sim.py --json
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

检查：

```text
deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

确认 manifest 里的 KModel、scaler 和 profile 是本次要上线的版本。

## 新增模型类型

新增模型时按这个顺序：

1. 在 `raw_cnn_pc/raw_cnn/models.py` 新增模型类。
2. 在 `models.py` 增加对应 `build_*_from_config()`。
3. 在 `normalize_model_type()` 中加入类型别名。
4. 在 `build_model_from_config()` 分发到新模型。
5. 增加或复制一份 `configs/infer/*.json` 和 `configs/export/*.json`。
6. 导出 KModel 后，更新 `raw_cnn_k230/configs/runtime.json`。

如果模型需要板端新增输入形状或输出槽位规则，再同步更新 `raw_cnn_k230/runtime/`。

## 对比和量化工具

常用辅助脚本：

```text
compare_pth_onnx_kmodel_gui.py
compare_pth_kmodel.py
compare_layerwise_pth_onnx.py
evaluate_quantization_schemes.py
evaluate_seg3_quant_schemes.py
run_pth_csv_compare_gui.py
```

这些脚本应继续通过 `infer.build_model_from_config()` 或 `build_kmodel.build_model_from_config()` 间接使用公共模型层。

## 依赖

基础推理依赖：

```bash
pip install -r requirements.txt
```

导出 KModel 依赖：

```bash
pip install -r requirements_k230_host.txt
```

如果 `nncase` 提示缺少 `.NET Runtime`，按当前 nncase 版本要求安装对应 .NET runtime。
