# Raw CNN K230 Folder

This folder is board deploy package.  
After PC export, copy this whole folder to SD card and run on K230.

For model replacement + export + deploy workflow, see:
- `../raw_cnn_pc/MODEL_CHANGE_TO_K230_GUIDE.md`

## Contains
- `run_k230_infer.py`: board-side inference script
- `k230_config.json`: runtime config
- `model/`: kmodel and scaler json
- `test_data/`: optional test csv data

## Run on board

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

## Quick test sample count

`k230_config.json` includes:

```json
"runtime": {
  "max_samples": 10
}
```

Set `max_samples` to a larger value for full test, or remove it to run all samples.

## UART output format

When `uart.enabled=true` in `k230_config.json`, each inference sends one frame to MCU:

- frame: `AA | 12 x int32 (little-endian) | FF`
- total length: `50 bytes`
- value conversion: `int(round(prediction * predict_scale))`
- current single-output model mapping:
  - channel 1: scaled prediction
  - channel 2..12: 0

Default UART config in this folder uses UART2 (`tx_pin=11`, `rx_pin=12`).
