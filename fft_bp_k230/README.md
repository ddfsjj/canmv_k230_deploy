# FFT BP K230 Folder

这个目录是板端部署包。  
PC 导出完成后，把这个目录整体拷到 SD 卡，在 K230 上运行。

完整替换与部署流程：
- `../fft_bp_pc/FFT_BP_PC_K230_完整流程说明.md`

## 包含内容

- `run_k230_infer.py`：板端推理脚本
- `k230_config.json`：板端配置
- `model/`：kmodel 与 scaler json
- `test_data/`：可选测试数据

## 板端运行

```python
cd /sdcard/fft_bp_k230
python run_k230_infer.py
```
