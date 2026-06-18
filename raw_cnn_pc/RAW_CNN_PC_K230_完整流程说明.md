# RAW CNN PC + K230 完整流程说明

这份文档只保留当前实际使用的主流程，避免在旧配置、旧脚本之间来回找。

当前主线是 `CNN-TCN`：

- PC 端目录：`raw_cnn_pc/`
- K230 板端目录：`raw_cnn_k230/`
- PC 推理配置：`raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json`
- PC 导出配置：`raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json`
- 板端运行配置：`raw_cnn_k230/configs/k230_config_cnn_tcn.json`
- PC 端生成 kmodel 入口：`raw_cnn_pc/build_kmodel.py`
- 板端运行入口：`raw_cnn_k230/run_k230_infer.py`

当前已经验证可上板的模型是：

```text
raw_cnn_pc/model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057/cnn_tcn.pth
raw_cnn_pc/model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057/scaler.pkl
raw_cnn_k230/model/cnn_tcn_20260415_074057_i16u8_kld512.kmodel
raw_cnn_k230/model/scaler_cnn_tcn_20260415_074057_i16u8_kld512.json
```

当前推荐量化方案：

```json
{
  "quant_type": "int16",
  "weight_quant_type": "uint8",
  "calibrate_method": "Kld",
  "samples_count": 512,
  "sampling_strategy": "per_dryness_uniform"
}
```

原因：`074057` 用 `NoClip` 时出现过固定大输出饱和，`Kld` 明显更稳。

## 1. 三个核心配置文件

只要跑当前 CNN-TCN 主线，优先只看这三份。

### 1.1 PC 推理配置

文件：

```text
raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json
```

作用：

- 控制 PC 端 `.pth` 推理。
- 决定用哪个 `.pth`。
- 决定用哪个 `scaler.pkl`。
- 决定测试数据目录、切窗参数、预处理方式。

最常改：

```json
{
  "data": {
    "test_data_dir": "data/880k_data_260414",
    "base_window_size": 500,
    "base_step": 200,
    "sequence_length": 5,
    "sequence_step": 1
  },
  "preprocessing": {
    "feature_mode": "window_demean"
  },
  "model": {
    "weights_path": "model/cnn-tcn/你的模型目录/cnn_tcn.pth"
  },
  "normalization": {
    "scaler_path": "model/cnn-tcn/你的模型目录/scaler.pkl"
  }
}
```

### 1.2 PC 导出配置

文件：

```text
raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json
```

作用：

- 控制从 `.pth` 导出到 `.onnx/.kmodel/scaler_json`。
- 控制量化校准数据来源。
- 控制量化方案。
- 控制导出文件名。

最常改：

```json
{
  "paths": {
    "weights_pth": "model/cnn-tcn/你的模型目录/cnn_tcn.pth",
    "scaler_pkl": "model/cnn-tcn/你的模型目录/scaler.pkl",
    "onnx": "../raw_cnn_k230/model/你的模型名.onnx",
    "kmodel": "../raw_cnn_k230/model/你的模型名.kmodel",
    "scaler_json": "../raw_cnn_k230/model/scaler_你的模型名.json",
    "calibration_npy": "../raw_cnn_k230/model/你的模型名_calibration_input.npy",
    "calibration_data_dir": "data/880k_data_260414",
    "test_data_dir": "data/880k_data_260414",
    "predictions_csv": "../raw_cnn_k230/predictions_你的模型名.csv",
    "nncase_dump_dir": "../raw_cnn_k230/model/nncase_dump_你的模型名"
  },
  "quantization": {
    "samples_count": 512,
    "sampling_strategy": "per_dryness_uniform",
    "random_seed": 20260414,
    "quant_type": "int16",
    "weight_quant_type": "uint8",
    "calibrate_method": "Kld"
  }
}
```

### 1.3 板端运行配置

文件：

```text
raw_cnn_k230/configs/k230_config_cnn_tcn.json
```

作用：

- 控制 K230 加载哪个 `kmodel`。
- 控制 K230 加载哪个 `scaler_json`。
- 控制离线 CSV 数据目录。
- 控制运行模式：离线 CSV 或在线 UART。
- 控制串口协议参数。

最常改：

```json
{
  "paths": {
    "kmodel": "model/你的模型名.kmodel",
    "scaler_json": "model/scaler_你的模型名.json",
    "test_data_dir": "data/880k_data_260414",
    "predictions_csv": "predictions_你的模型名.csv"
  },
  "runtime": {
    "mode": "csv_cached"
  }
}
```

注意：`kmodel` 和 `scaler_json` 必须来自同一次导出，不能混用。

## 2. 新模型换上板流程

如果 CNN-TCN 网络结构不变，只换新的 `.pth/.pkl`，按这一节做。

### 第 1 步：放入新模型

建议每个模型单独一个目录，例如：

```text
raw_cnn_pc/model/cnn-tcn/你的模型目录/cnn_tcn.pth
raw_cnn_pc/model/cnn-tcn/你的模型目录/scaler.pkl
raw_cnn_pc/model/cnn-tcn/你的模型目录/cnn_tcn.meta.json
```

### 第 2 步：改 PC 推理配置

改：

```text
raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json
```

只换模型时，重点改这两处：

```json
"weights_path": "model/cnn-tcn/你的模型目录/cnn_tcn.pth"
```

```json
"scaler_path": "model/cnn-tcn/你的模型目录/scaler.pkl"
```

### 第 3 步：改 PC 导出配置

改：

```text
raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json
```

至少确认这些路径都换成同一个模型名：

```json
"weights_pth": "model/cnn-tcn/你的模型目录/cnn_tcn.pth",
"scaler_pkl": "model/cnn-tcn/你的模型目录/scaler.pkl",
"onnx": "../raw_cnn_k230/model/你的模型名.onnx",
"kmodel": "../raw_cnn_k230/model/你的模型名.kmodel",
"scaler_json": "../raw_cnn_k230/model/scaler_你的模型名.json",
"calibration_npy": "../raw_cnn_k230/model/你的模型名_calibration_input.npy",
"nncase_dump_dir": "../raw_cnn_k230/model/nncase_dump_你的模型名"
```

文件名建议带上：

- 模型时间或版本号。
- 量化类型。
- 校准方法。
- 校准样本数。

例如：

```text
cnn_tcn_20260415_074057_i16u8_kld512.kmodel
scaler_cnn_tcn_20260415_074057_i16u8_kld512.json
```

### 第 4 步：生成 kmodel

在 PC 上运行：

```powershell
cd raw_cnn_pc
..\.venv\Scripts\python.exe build_kmodel.py
```

等价于显式指定当前默认导出配置：

```powershell
..\.venv\Scripts\python.exe build_kmodel.py --config configs/export/k230_export_config_cnn_tcn.json
```

成功时会看到：

```text
Exported ONNX: ...
Exported scaler json: ...
Saved calibration data: ...
Generated kmodel: ...
```

生成结果在：

```text
raw_cnn_k230/model/
```

至少需要：

```text
你的模型名.kmodel
scaler_你的模型名.json
```

### 第 5 步：PC 端验证 kmodel

不要生成完就直接上板，先在 PC 上对比 PTH / ONNX / KMODEL。

```powershell
cd raw_cnn_pc
..\.venv\Scripts\python.exe compare_pth_onnx_kmodel_gui.py
  --summary_json artifacts/reports/你的模型名_summary.json `
  --details_csv artifacts/reports/你的模型名_details.csv `
  --per_csv_csv artifacts/reports/你的模型名_per_csv.csv `
  --per_dryness_csv artifacts/reports/你的模型名_per_dryness.csv
```

重点看：

```text
pth_mae_vs_true
onnx_mae_vs_true
kmodel_mae_vs_true
pth_vs_onnx_mae
pth_vs_kmodel_mae
pth_vs_kmodel_max_abs
```

判断：

- `pth_vs_onnx_mae` 应该接近 0，通常是 `1e-7` 级别。
- `kmodel_mae_vs_true` 明显变大，说明量化后不稳定。
- 多个样本输出同一个很大的固定值，通常是量化饱和。
- 饱和时优先把 `calibrate_method` 改成 `Kld`。

当前 `074057_i16u8_kld512` 的验证结果：

```text
pth_mae_vs_true: 0.037184
onnx_mae_vs_true: 0.037184
kmodel_mae_vs_true: 0.116889
pth_vs_kmodel_mae: 0.103242
```

### 第 6 步：改板端配置

改：

```text
raw_cnn_k230/configs/k230_config_cnn_tcn.json
```

把路径指向新生成的板端文件：

```json
"paths": {
  "kmodel": "model/你的模型名.kmodel",
  "scaler_json": "model/scaler_你的模型名.json",
  "test_data_dir": "data/880k_data_260414",
  "predictions_csv": "predictions_你的模型名.csv"
}
```

## 3. 板端怎么跑

板端目录建议保持：

```text
/sdcard/raw_cnn_k230/
```

至少需要拷贝：

```text
/sdcard/raw_cnn_k230/run_k230_infer.py
/sdcard/raw_cnn_k230/run_k230_csv_compare.py
/sdcard/raw_cnn_k230/configs/k230_config_cnn_tcn.json
/sdcard/raw_cnn_k230/model/你的模型名.kmodel
/sdcard/raw_cnn_k230/model/scaler_你的模型名.json
```

如果跑离线 CSV，还需要：

```text
/sdcard/raw_cnn_k230/data/880k_data_260414/
```

如果跑在线 UART，不需要 CSV 数据目录。

### 3.1 离线 CSV 测试

配置：

```json
"runtime": {
  "mode": "csv_cached"
}
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

输出：

- 每条样本预测值。
- 总体 MAE / RMSE。
- 预测 CSV。

输出 CSV 文件名会自动带上当前 kmodel 名，避免多次测试混在一起。

### 3.2 离线 CSV 对比工具

如果想要更偏“对比分析”的输出，运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

它和 `run_k230_infer.py` 的区别：

- `run_k230_infer.py` 是正式入口，按 `runtime.mode` 跑。
- `run_k230_csv_compare.py` 是离线分析工具，会输出更多按 CSV 汇总的对比结果。

### 3.3 在线 UART 测试

配置：

```json
"runtime": {
  "mode": "uart_online"
}
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

在线模式只依赖：

- `kmodel`
- `scaler_json`
- 串口配置

不依赖：

- `test_data_dir`
- CSV 测试数据

## 4. 离线和在线模式区别

### `csv_cached`

用途：

- 板端直接读 CSV。
- 验证 kmodel 在板端跑出来是否正常。
- 不依赖上位机串口发数据。

需要：

```text
/sdcard/raw_cnn_k230/data/880k_data_260414/
```

### `uart_online`

用途：

- 实际在线推理。
- 从串口收实时数据。
- 达到窗口长度后自动推理并回传。

不需要 CSV 数据。

需要确认：

- `runtime.uart_online.channel_count`
- `runtime.uart_online.input_value_type`
- `runtime.uart_online.input_byte_order`
- `runtime.uart_online.infer_step_frames`
- `uart.predict_scale`
- `uart.value_type`
- `uart.byte_order`
- `uart.value_count`
- `uart.header`
- `uart.tail`
- 如果启用外层大帧，还要确认 `outer_*` 参数。

## 5. 串口联调完整步骤

串口联调不要一开始就跑 `uart_online`。正确顺序是从“最不依赖模型”的模式开始，一层一层加功能。

推荐顺序：

```text
1. uart_continuous_send_test.py
2. uart_echo
3. uart_frame_return
4. uart_debug_ack
5. uart_online
```

每一步通过后再进入下一步。

### 5.1 第一步：`uart_continuous_send_test.py`

作用：

- 只测试 K230 串口发送能力。
- 不加载 kmodel。
- 不读取 CSV。
- 不解析上位机数据。
- 用来确认 K230 的 TX 引脚、波特率、帧格式是否正确。

运行文件：

```text
raw_cnn_k230/uart_continuous_send_test.py
```

板端运行：

```python
cd /sdcard/raw_cnn_k230
python uart_continuous_send_test.py
```

当前脚本默认发送小帧：

```text
55 AA + 12 个 4 字节整数 + FC CF
```

默认参数：

```text
UART_ID = 2
TX_PIN = 11
RX_PIN = 12
BAUDRATE = 921600
VALUE_COUNT = 12
BYTE_ORDER = big
```

每帧长度：

```text
2 + 12 * 4 + 2 = 52 bytes
```

能验证什么：

- K230 TX 引脚是否接对。
- 上位机 RX 是否能收到数据。
- 波特率是否一致。
- 小帧头 `55 AA`、帧尾 `FC CF` 是否一致。
- 12 路数值是否能按大端 int32 正确解析。

预期现象：

- 上位机持续收到 52 字节小帧。
- 第一项数值是递增序号。
- 第二项数值是板端毫秒时间。

如果这一步不通：

- 先不要看模型。
- 优先查 TX/RX 是否接反。
- 查 UART ID、引脚、波特率。
- 查上位机是否按大端 int32 解码。

### 5.2 第二步：`uart_echo`

作用：

- 测试 K230 串口收发闭环。
- K230 收到什么原样发回什么。
- 不解析帧头帧尾。
- 不判断帧长度。
- 不加载模型。

适合验证：

- 上位机发到 K230 是否通。
- K230 回到上位机是否通。
- 双向串口链路是否正常。

配置文件：

```text
raw_cnn_k230/configs/k230_config_cnn_tcn.json
```

把模式改成：

```json
"runtime": {
  "mode": "uart_echo"
}
```

相关配置：

```json
"uart_echo": {
  "idle_sleep_ms": 1,
  "log_every_n_packets": 20,
  "print_hex": false
}
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

它会做什么：

- 不管收到的是不是合法帧。
- 每次 `uart.read()` 读到一段 bytes，就直接 `uart.write()` 原样写回。
- 不改任何字节。

预期日志：

```text
uart_echo_start
uart_echo_cfg
uart_echo_stat: packets=..., rx_bytes=..., tx_bytes=...
```

如果 `print_hex = true`，会打印每次收到的原始字节：

```text
uart_echo_packet: bytes=... hex=...
```

通过标准：

- 上位机发什么，能收到完全一样的字节。
- `rx_bytes` 和 `tx_bytes` 持续增加。
- 不要求帧格式正确。

如果这一步不通：

- 说明双向链路还没通。
- 查 RX/TX 是否接反。
- 查上位机发送是否真的发出。
- 查 K230 `uart.enabled` 是否为 `true`。
- 查 `uart_id/tx_pin/rx_pin/baudrate`。

### 5.3 第三步：`uart_frame_return`

作用：

- 测试 K230 是否能按协议正确收完整帧。
- K230 每收到 N 帧，回传其中一帧。
- 不加载模型。
- 不做推理。

适合验证：

- 小帧格式是否正确。
- 大帧格式是否正确。
- 帧头帧尾是否对齐。
- K230 是否能从字节流里拆出完整帧。

配置：

```json
"runtime": {
  "mode": "uart_frame_return",
  "uart_frame_return": {
    "return_every_n_frames": 1,
    "idle_sleep_ms": 1,
    "log_every_n_frames": 100,
    "print_hex": false,
    "strict_protocol": true,
    "fixed_frame_len": 1044,
    "return_inner_frame_when_outer_enabled": true,
    "return_inner_frame_index": -1
  }
}
```

当前板端大帧配置：

```json
"uart": {
  "header": [85, 170],
  "tail": [252, 207],
  "outer_frame_enabled": true,
  "outer_frame_count": 20,
  "outer_header": [247, 127],
  "outer_tail": [250, 175]
}
```

小帧格式：

```text
55 AA + 12 * int32 + FC CF
```

小帧长度：

```text
52 bytes
```

大帧格式：

```text
F7 7F + 20 个小帧 + FA AF
```

大帧长度：

```text
2 + 20 * 52 + 2 = 1044 bytes
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

它会做什么：

- 如果 `strict_protocol = true`，按帧头帧尾解析。
- 如果 `outer_frame_enabled = true`，先找完整大帧，再检查大帧里的 20 个小帧。
- 每收到 `return_every_n_frames` 个完整大帧，就回传一帧。
- 当前 `return_inner_frame_when_outer_enabled = true`，所以收到大帧后回传其中一个小帧。
- `return_inner_frame_index = -1` 表示回传大帧里的最后一个小帧。

预期日志：

```text
uart_frame_return_start
uart_frame_return_cfg
uart_frame_return_outer_frame_cfg
uart_frame_return_hit: rx_frame_idx=..., tx_frames=..., rx_bytes=..., tx_bytes=...
```

通过标准：

- 上位机持续发大帧。
- K230 的 `rx_frame_idx` 持续增加。
- 上位机能收到 K230 回传的小帧。
- 回传小帧内容能和上位机发送的大帧中某个小帧对上。

如果这一步不通：

- 如果 `rx_bytes` 增加但 `rx_frame_idx` 不增加，说明字节到了，但协议解析失败。
- 查大帧头是否是 `F7 7F`。
- 查大帧尾是否是 `FA AF`。
- 查每个小帧头是否是 `55 AA`。
- 查每个小帧尾是否是 `FC CF`。
- 查大帧里是否刚好 20 个小帧。
- 查上位机发送字节长度是否是 1044。

什么时候用 `strict_protocol = false`：

- 只想按固定长度切包，不想检查帧头帧尾。
- 排查帧头帧尾不确定的问题。

一般正式联调用 `strict_protocol = true`。

### 5.4 第四步：`uart_debug_ack`

作用：

- K230 每收到 1 个完整大帧，就立即回 1 个 ACK 小帧。
- 不加载模型。
- 不做推理。
- 用来确认大帧到达频率、计数、回传时序。

和 `uart_frame_return` 的区别：

- `uart_frame_return` 回传收到的原始帧内容。
- `uart_debug_ack` 回传 K230 自己生成的调试 ACK，里面带计数和时间戳。

配置：

```json
"runtime": {
  "mode": "uart_debug_ack",
  "uart_debug_ack": {
    "idle_sleep_ms": 1,
    "log_every_n_frames": 20,
    "print_hex": false,
    "strict_protocol": true,
    "fixed_frame_len": 1044,
    "flush_rx_on_start": true,
    "startup_flush_empty_rounds": 3,
    "startup_flush_sleep_ms": 10,
    "ack_magic": 9001
  }
}
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

ACK 小帧内容是 12 个 int32：

```text
[0] ack_magic，默认 9001
[1] ack_seq，ACK 序号，从 1 递增
[2] board_ticks_ms，板端毫秒时间
[3] rx_outer_frame_idx，收到的完整大帧序号
[4] rx_small_frame_idx，折算后的小帧序号
[5] total_rx_bytes，累计接收字节数
[6] len(frame)，本次完整帧长度，当前大帧应为 1044
[7] 固定 1，表示成功收到完整帧
[8] 预留
[9] 预留
[10] 预留
[11] 预留
```

ACK 帧仍然按普通小帧格式发回：

```text
55 AA + 12 * int32 + FC CF
```

预期日志：

```text
uart_debug_ack_start
uart_debug_ack_cfg
uart_debug_ack_outer_frame_cfg
uart_debug_ack_startup_flush
uart_debug_ack_stat: rx_outer_frames=..., rx_small_frames=..., tx_ack_frames=..., rx_bytes=..., ack_seq=...
```

通过标准：

- 上位机每发 1 个完整大帧，能收到 1 个 ACK 小帧。
- ACK 第 1 个值是 `9001`。
- ACK 第 2 个值连续递增。
- ACK 第 7 个值是 `1044`。
- ACK 频率和上位机发大帧频率一致。

如果这一步不通：

- 如果没有 ACK，但 `uart_frame_return` 正常，查 ACK 解码方式。
- 如果 ACK 序号跳变，查上位机发包是否丢包。
- 如果 `len(frame)` 不是 1044，查大帧长度。
- 如果启动后先出现异常 ACK，保留 `flush_rx_on_start = true` 清空旧缓存。

### 5.5 第五步：`uart_online`

作用：

- 正式在线推理模式。
- 从 UART 接收实时 12 路数据。
- 按窗口参数缓存数据。
- 满足模型输入长度后运行 kmodel。
- 把 12 路预测值按串口协议发回。

配置：

```json
"runtime": {
  "mode": "uart_online",
  "uart_online": {
    "channel_count": 12,
    "infer_step_frames": 200,
    "input_value_type": "int32",
    "input_byte_order": "big",
    "idle_sleep_ms": 1,
    "log_every_n_frames": 200,
    "send_zeros_before_ready": false,
    "debug_predict_trace": true,
    "debug_outer_rx": true,
    "flush_rx_on_start": true
  }
}
```

运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

它会做什么：

1. 加载 `kmodel`。
2. 加载 `scaler_json`。
3. 初始化 UART。
4. 解析上位机发来的小帧或大帧。
5. 每个小帧得到 12 路原始值。
6. 每一路单独维护长度为 `base_window_size` 的滑动窗口。
7. 每隔 `base_step` 生成一个基础窗口。
8. CNN-TCN 会继续收集 `sequence_length` 个基础窗口。
9. 满足 `sequence_step` 后推理一次。
10. 每次推理输出 12 路预测值并回传。

当前 CNN-TCN 数据参数：

```json
"data": {
  "base_window_size": 500,
  "base_step": 200,
  "sequence_length": 5,
  "sequence_step": 1
}
```

含义：

- 每一路先攒满 500 帧原始数据。
- 之后每 200 帧生成一个基础窗口。
- CNN-TCN 需要连续 5 个基础窗口。
- 所以第一次正式推理需要大约：

```text
500 + 200 * (5 - 1) = 1300 个小帧
```

当前大帧每个包含 20 个小帧，所以第一次推理大约需要：

```text
1300 / 20 = 65 个大帧
```

之后每 `base_step = 200` 个小帧推理一次，也就是大约每：

```text
200 / 20 = 10 个大帧
```

回传格式：

```text
55 AA + 12 * int32 + FC CF
```

其中预测值会按：

```json
"predict_scale": 10000
```

放大后转成 int32。

例如预测 `0.1234`，回传整数约为：

```text
1234
```

预期日志：

```text
uart_online_start
uart_online_cfg
uart_online_feature_mode
uart_online_outer_frame_cfg
uart_online_startup_flush
uart_online_outer_rx
uart_online_trigger
uart_online_result
uart_online_tx
```

关键日志含义：

- `uart_online_outer_rx`：K230 收到了大帧并拆出小帧。
- `uart_online_trigger`：缓存已经满足条件，准备推理。
- `uart_online_result`：已经完成一次 kmodel 推理。
- `uart_online_tx`：已经把预测结果发回上位机。

通过标准：

- `uart_online_outer_rx` 持续出现，说明接收正常。
- 第一次缓存满后出现 `uart_online_trigger`。
- 随后出现 `uart_online_result`。
- 上位机能收到 52 字节结果小帧。
- 结果小帧能按 `predict_scale = 10000` 还原成浮点预测值。

如果这一步不通：

- 有 `outer_rx` 但没有 `trigger`：说明帧收到了，但还没攒够 1300 个小帧，继续等。
- 长时间没有 `outer_rx`：说明协议解析失败或串口没收到。
- 有 `trigger` 但没有 `result`：重点查 kmodel/scaler 是否正确。
- 有 `result` 但上位机收不到：查 K230 TX、上位机 RX、返回帧解析。
- 结果全是 0：查 `send_zeros_before_ready`，或者确认是否还没到第一次推理。

### 5.6 各模式功能对比

| 模式 | 是否需要上位机发数据 | 是否解析协议 | 是否加载模型 | 是否推理 | 回传内容 | 用途 |
|---|---|---|---|---|---|---|
| `uart_continuous_send_test.py` | 否 | 否 | 否 | 否 | 固定测试小帧 | 测 K230 发送和上位机接收 |
| `uart_echo` | 是 | 否 | 否 | 否 | 收到什么回什么 | 测双向串口链路 |
| `uart_frame_return` | 是 | 是 | 否 | 否 | 原始帧或大帧里的小帧 | 测帧格式和大小帧解析 |
| `uart_debug_ack` | 是 | 是 | 否 | 否 | ACK 小帧 | 测大帧到达频率和回传时序 |
| `uart_online` | 是 | 是 | 是 | 是 | 12 路预测值 | 正式在线推理 |

### 5.7 当前协议速查

小帧：

```text
55 AA + 12 * int32(big-endian) + FC CF
```

小帧长度：

```text
52 bytes
```

大帧：

```text
F7 7F + 20 * 小帧 + FA AF
```

大帧长度：

```text
1044 bytes
```

输入：

```json
"input_value_type": "int32",
"input_byte_order": "big"
```

输出：

```json
"value_type": "int32",
"byte_order": "big",
"predict_scale": 10000
```

### 5.8 推荐排查路径

如果串口不通：

```text
uart_continuous_send_test.py -> uart_echo -> uart_frame_return -> uart_debug_ack -> uart_online
```

如果能收到但模型不出结果：

```text
uart_frame_return -> uart_debug_ack -> uart_online
```

如果离线 CSV 正常但在线结果不正常：

```text
重点查 input_value_type、input_byte_order、predict_scale、帧长度、大小帧数量
```

如果在线很久没有第一次结果：

```text
确认是否已经收到至少 1300 个小帧，当前等价于约 65 个大帧
```

## 6. 常见问题排查

### 6.1 `load_state_dict` 报错

通常是 `.pth` 和配置里的模型结构不匹配。

检查：

- `model.type`
- `cnn_tcn_conv_filters`
- `cnn_tcn_kernel_size`
- `cnn_tcn_pool_size`
- `cnn_tcn_num_channels`
- `cnn_tcn_tcn_kernel_size`
- `cnn_tcn_dilations`
- `sequence_length`

### 6.2 `No valid samples`

通常是数据切不出样本。

检查：

- `test_data_dir` 是否对。
- CSV 是否足够长。
- `base_window_size` 是否太大。
- `sequence_length` 是否太大。
- `base_step` / `sequence_step` 是否导致切不出序列。

### 6.3 `nncase is not installed`

通常是 Python 环境不对。

建议在 PC 端使用项目虚拟环境：

```powershell
cd raw_cnn_pc
..\.venv\Scripts\python.exe build_kmodel.py
```

如果还缺依赖，安装：

```powershell
pip install -r requirements_k230_host.txt
```

### 6.4 kmodel 结果明显不对

优先检查三边一致性：

- PC 推理配置里的 `data.*`
- PC 导出配置里的 `data.*`
- 板端配置里的 `data.*`
- `feature_mode`
- `scaler.pkl` 和 `scaler_json` 是否配套。
- `kmodel` 和 `scaler_json` 是否同一次导出。

如果出现固定大输出，例如大量样本都输出 `9.3671875`：

- 优先把 `calibrate_method` 从 `NoClip` 改成 `Kld`。
- 其次试 `quant_type = int16`、`weight_quant_type = uint8`。
- 再重新生成 kmodel 并跑 PC 对比。

### 6.5 板端输出文件分不清

现在板端输出 CSV 会自动追加 kmodel 文件名。

例如配置里写：

```json
"predictions_csv": "predictions_cnn_tcn.csv"
```

如果当前 kmodel 是：

```text
cnn_tcn_20260415_074057_i16u8_kld512.kmodel
```

实际输出会带上：

```text
predictions_cnn_tcn__cnn_tcn_20260415_074057_i16u8_kld512.csv
```

## 7. 最短执行版

如果只是换了 CNN-TCN 新 `.pth/.pkl`，结构没变，最短按下面做：

1. 把新 `.pth/.pkl` 放到 `raw_cnn_pc/model/cnn-tcn/你的模型目录/`。
2. 改 `raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json` 的 `weights_path` 和 `scaler_path`。
3. 改 `raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json` 的 `weights_pth` 和 `scaler_pkl`。
4. 改导出文件名：`onnx/kmodel/scaler_json/calibration_npy/nncase_dump_dir`。
5. 量化方案优先用 `int16 + uint8 + Kld`。
6. 运行 `cd raw_cnn_pc`。
7. 运行 `..\.venv\Scripts\python.exe build_kmodel.py`。
8. 运行 PC 对比脚本确认 kmodel 不离谱。
9. 改 `raw_cnn_k230/configs/k230_config_cnn_tcn.json` 的 `paths.kmodel` 和 `paths.scaler_json`。
10. 把 `kmodel/scaler_json/config` 拷到板端。
11. 离线测：`runtime.mode = "csv_cached"`，运行 `python run_k230_infer.py`。
12. 在线测：`runtime.mode = "uart_online"`，运行 `python run_k230_infer.py`。

## 8. 当前 074057 可上板配置

当前 PC 导出配置已经指向：

```text
raw_cnn_pc/model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057/cnn_tcn.pth
raw_cnn_pc/model/cnn-tcn/train_model_bundle_cnn_tcn_20260415_074057/scaler.pkl
```

当前板端配置已经指向：

```json
"paths": {
  "kmodel": "model/cnn_tcn_20260415_074057_i16u8_kld512.kmodel",
  "scaler_json": "model/scaler_cnn_tcn_20260415_074057_i16u8_kld512.json",
  "test_data_dir": "data/880k_data_260414",
  "predictions_csv": "predictions_cnn_tcn_k230_20260415_074057_i16u8_kld512.csv"
}
```

所以上板前确认板子上有：

```text
/sdcard/raw_cnn_k230/model/cnn_tcn_20260415_074057_i16u8_kld512.kmodel
/sdcard/raw_cnn_k230/model/scaler_cnn_tcn_20260415_074057_i16u8_kld512.json
```

如果是离线 CSV 测试，再确认：

```text
/sdcard/raw_cnn_k230/data/880k_data_260414/
```
