# FFT BP PC Folder

这个目录用于 PC 侧工作：
- 运行 `.pth` 推理（`infer.py`）
- 将 `.pth` 导出为 K230 侧可用产物（`build_kmodel.py`）
- 导出产物自动写入 `../fft_bp_k230`

详细流程文档：
- `FFT_BP_PC_K230_完整流程说明.md`

## 1) 安装依赖

```bash
pip install -r requirements.txt
pip install -r requirements_k230_host.txt
```

## 2) PC 推理

```bash
python infer.py --config infer_config.json --output predictions.csv
```

## 3) 导出到 K230 目录

```bash
python build_kmodel.py
```

或

```bash
build_to_k230.bat
```

默认导出位置：
- `../fft_bp_k230/model/model_run000.onnx`
- `../fft_bp_k230/model/model_run000.kmodel`
- `../fft_bp_k230/model/scaler_run000.json`
- `../fft_bp_k230/model/calibration_input.npy`
