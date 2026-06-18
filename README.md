# Raw CNN K230 使用说明

这个目录是 K230 板端运行目录，当前默认已经切到 `CNN-TCN` 在线推理配置。

建议板端目录结构保持为：

```text
/sdcard/raw_cnn_k230/
```

## 当前默认配置

当前 [k230_config.json](/d:/code/network/canmv_k230_deploy/raw_cnn_k230/k230_config.json:1) 默认指向：

- `model.type = "cnn_tcn"`
- `paths.kmodel = "model/cnn_tcn_20260414_013210.kmodel"`
- `paths.scaler_json = "model/scaler_cnn_tcn_20260414_013210.json"`
- `runtime.mode = "uart_online"`
- `uart.enabled = true`

也就是说，目录拷到板子后，默认目标就是直接跑 `CNN-TCN` 串口在线推理。

## 目录里最重要的文件

- `k230_config.json`
  板端实际运行配置，改模型、改模式、改串口都从这里下手。
- `k230_config.annotated.jsonc`
  带注释版配置，适合查字段用途。
- `run_k230_infer.py`
  板端统一入口，支持离线推理、串口在线推理、串口联调模式。
- `run_k230_csv_compare.py`
  板端离线 CSV 对比入口。
- `boot.py` / `main.py`
  如果板子配置成上电自启，通常从这里进入。
- `model/*.kmodel`
  板端加载的模型文件。
- `model/*.json`
  板端使用的标准化参数。

## 推荐使用的配置目录

为了把三种模型的配置分开，当前建议优先使用下面这 9 份配置：

- PC 推理：
  `raw_cnn_pc/configs/infer/infer_config_cnn.json`
  `raw_cnn_pc/configs/infer/infer_config_cnn_lstm.json`
  `raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json`
- 导出：
  `raw_cnn_pc/configs/export/k230_export_config_cnn.json`
  `raw_cnn_pc/configs/export/k230_export_config_cnn_lstm.json`
  `raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json`
- 板端运行：
  `raw_cnn_k230/configs/k230_config_cnn.json`
  `raw_cnn_k230/configs/k230_config_cnn_lstm.json`
  `raw_cnn_k230/configs/k230_config_cnn_tcn.json`

板端脚本现在支持显式传配置路径，例如：

```python
python run_k230_infer.py configs/k230_config_cnn_tcn.json
python run_k230_csv_compare.py configs/k230_config_cnn_tcn.json 10
```

## CNN-TCN 结构说明

当前这版 `CNN-TCN` 不是单窗模型，而是“窗口内 CNN + 序列间 TCN”的两段式结构。

输入样本形状：

```text
(batch, sequence_length, base_window_size) = (batch, 5, 500)
```

也就是说，每次推理输入由最近 `5` 个基础窗口组成，每个基础窗口长度为 `500`。

### 1. 窗口内 CNN 编码

每个基础窗口先单独过一套 `1D CNN`：

- `Conv1d(1 -> 16, kernel_size=5, padding=2)` + `ReLU` + `MaxPool1d(2)`
- `Conv1d(16 -> 32, kernel_size=3, padding=1)` + `ReLU` + `MaxPool1d(2)`
- 对卷积输出做 `mean(dim=-1)`，把单个窗口压成 `32` 维特征

所以一个 `(5, 500)` 的序列样本，先会变成 `(5, 32)`。

### 2. 序列间 TCN 建模

然后把这 `5` 个窗口特征按时间顺序送进 TCN：

- `TemporalBlock(32 -> 32, kernel_size=3, dilation=1)`
- `TemporalBlock(32 -> 32, kernel_size=3, dilation=2)`
- `TemporalBlock(32 -> 48, kernel_size=3, dilation=4)`

每个 `TemporalBlock` 都包含：

- 两层 `CausalConv1d`
- `ReLU`
- `Dropout(0.0)`
- 残差连接
- 当输入输出通道数不一致时，用 `1x1 Conv` 对齐残差支路

### 3. 回归输出头

TCN 输出后：

- 取最后一个时间步特征
- 通过 `Linear(48 -> 1)`
- 输出 1 个回归值

## 板端在线推理的数据流

当前默认在线模式是 `runtime.mode = "uart_online"`，并且对 `cnn_tcn` 走的是“序列模型在线逻辑”。

### 输入数据含义

单片机每发来 1 个“小帧”，板端会解析出 `12` 路数值：

```text
1 小帧 = 12 路通道在同一个时刻的采样值
```

当前默认：

- `runtime.uart_online.channel_count = 12`
- `uart.value_count = 12`

所以每次串口接收到 1 帧，就相当于收到了 12 路通道的新时间点。

### 在线缓存与触发逻辑

当前关键参数：

- `base_window_size = 500`
- `base_step = 200`
- `sequence_length = 5`
- `sequence_step = 1`

在线推理逻辑分 4 层：

1. 先持续接收小帧，写入每个通道自己的长度为 `500` 的环形缓冲区。
2. 当环形缓冲区第一次装满后，才具备生成 1 个基础窗口的条件。
3. 之后每累计 `200` 个新小帧，就生成 1 个新的基础窗口特征。
4. 当基础窗口累计到 `5` 个后，组成一个 `(5, 500)` 序列样本触发推理。
5. 因为 `sequence_step = 1`，后续每新增 `1` 个基础窗口，就再次触发一次推理。

### 从原始小帧到第一次出结果，需要多少数据

第一次推理至少需要：

- 先收满第 1 个基础窗口：`500` 帧
- 再额外生成后面 4 个基础窗口：`4 * 200 = 800` 帧

所以首次触发推理大约需要：

```text
500 + 800 = 1300 个小帧
```

如果单片机还开了外层“大帧”打包，当前默认：

- `uart.outer_frame_enabled = true`
- `uart.outer_frame_count = 20`

那么：

```text
1 个大帧 = 20 个小帧
```

也就是说第一次出结果大约对应：

```text
1300 / 20 = 65 个大帧
```

这只是首次结果。之后每新增 `200` 个小帧，也就是每 `10` 个大帧左右，会再触发一轮新基础窗口；而由于 `sequence_step = 1`，每次新基础窗口都会触发一次新的序列推理。

## 运行方式

## 1. 板端在线推理

当前默认就是这个模式。

在板端执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

如果配置无误，启动日志里应该能看到类似信息：

```text
uart_online_cfg: model_type=cnn_tcn, ...
```

如果这里显示的不是 `cnn_tcn`，说明您实际跑的配置文件不是当前这份。

## 2. 板端离线 CSV 对比

如果想先在板端验证模型本身，再联调串口，可以切到：

```json
"runtime": {
  "mode": "csv_cached"
}
```

然后运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

常见用法：

```python
python run_k230_csv_compare.py 10
python run_k230_csv_compare.py all
```

## 3. 串口联调模式

如果不要一上来就跑模型，可以按下面顺序联调：

1. `uart_continuous_send_test.py`
   先验证发送链路是否通。
2. `runtime.mode = "uart_echo"`
   收到什么回什么，先确认收发通道没问题。
3. `runtime.mode = "uart_frame_return"`
   验证按帧回传逻辑。
4. `runtime.mode = "uart_debug_ack"`
   验证帧计数、时间戳和 ACK 时序。
5. `runtime.mode = "uart_online"`
   最后再切到真实模型推理。

## 当前串口协议默认值

当前默认配置如下：

- `uart.uart_id = 2`
- `uart.tx_pin = 11`
- `uart.rx_pin = 12`
- `uart.baudrate = 921600`
- `uart.bits = 8`
- `uart.parity = "none"`
- `uart.stop = 1`
- `uart.value_type = "int32"`
- `uart.byte_order = "big"`
- `uart.predict_scale = 10000`
- `uart.header = [85, 170]`
- `uart.tail = [252, 207]`
- `uart.outer_frame_enabled = true`
- `uart.outer_frame_count = 20`
- `uart.outer_header = [247, 127]`
- `uart.outer_tail = [250, 175]`

### 小帧格式

当前每个小帧结构是：

```text
header + 12 * 4字节payload + tail
```

按默认值计算：

- 帧头 2 字节
- 12 个 `int32`，共 `48` 字节
- 帧尾 2 字节

所以：

```text
1 个小帧 = 52 字节
```

### 大帧格式

当前大帧结构是：

```text
outer_header + 20个小帧 + outer_tail
```

按默认值计算：

```text
2 + 20 * 52 + 2 = 1044 字节
```

这也对应了配置里的：

- `runtime.uart_frame_return.fixed_frame_len = 1044`
- `runtime.uart_debug_ack.fixed_frame_len = 1044`

## 最常改的配置项

### 1. 切模型文件

```json
"paths": {
  "kmodel": "model/xxx.kmodel",
  "scaler_json": "model/xxx.json"
}
```

注意 `kmodel` 和 `scaler_json` 必须成套。

### 2. 切运行模式

```json
"runtime": {
  "mode": "uart_online"
}
```

可选值：

- `uart_online`
- `csv_cached`
- `uart_echo`
- `uart_frame_return`
- `uart_debug_ack`

### 3. 调整在线触发节奏

```json
"data": {
  "base_window_size": 500,
  "base_step": 200,
  "sequence_length": 5,
  "sequence_step": 1
}
```

对 `cnn_tcn` 而言：

- `base_window_size` 决定单个基础窗口长度
- `base_step` 决定隔多少新小帧生成一个新基础窗口
- `sequence_length` 决定每次推理使用多少个基础窗口
- `sequence_step` 决定序列满后，新增多少个基础窗口再触发一次推理

### 4. 调整离线对比样本数

```json
"runtime": {
  "csv_cached": {
    "compare_max_samples": 10
  }
}
```

设置为 `null` 表示全量跑。

## 结果不对时优先检查什么

优先查下面几项：

1. `kmodel` 和 `scaler_json` 是否配套。
2. 板端 `data.*` 是否与 PC 导出配置一致，尤其是：
   `base_window_size`、`base_step`、`sequence_length`、`sequence_step`
3. `feature_mode` 是否一致，当前默认是 `window_demean`。
4. 串口协议是否一致，尤其是：
   `value_type`、`byte_order`、`header/tail`、`outer_frame_count`
5. 单片机送给板端的通道数是否真的是 `12` 路。
6. 联调时是否误用了旧的 `cnn_lstm` 配置文件或旧的模型文件。

## 建议的排查顺序

如果大将军上板后结果不出来，建议按这个顺序查：

1. 看启动日志里 `model_type` 是否是 `cnn_tcn`
2. 看 `kmodel` 和 `scaler_json` 是否是 `20260414_013210` 这套
3. 看单片机发包长度是否真的是 `1044` 字节大帧
4. 看板端是否已经收到足够多的小帧用于首次推理
5. 先切 `uart_echo` / `uart_debug_ack` 确认串口链路
6. 最后再回到 `uart_online`

## 所有上板流程总览

下面把当前仓库里实际可走通的上板流程按模型类型和目标场景统一列出来。

### 流程 A：当前默认 `CNN-TCN` 直接上板在线跑

适用场景：

- 现在就想跑仓库默认这版模型
- 不想先切回旧配置

当前对应配置：

- PC 推理配置：
  [infer_config_cnn_tcn.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/infer/infer_config_cnn_tcn.json:1)
- 导出配置：
  [k230_export_config_cnn_tcn.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json:1)
- 板端运行配置：
  [k230_config.json](/d:/code/network/canmv_k230_deploy/raw_cnn_k230/k230_config.json:1)

步骤：

1. 在 PC 端先确认 `configs/infer/infer_config_cnn_tcn.json` 能正常跑 `.pth`
2. 用 `configs/export/k230_export_config_cnn_tcn.json` 导出 `onnx/kmodel/scaler_json`
3. 确认板端 `k230_config.json` 仍然是：
   `model.type = "cnn_tcn"`
   `runtime.mode = "uart_online"`
4. 把整个 `raw_cnn_k230/` 拷到板端 `/sdcard/raw_cnn_k230/`
5. 板端执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

启动后重点看日志：

```text
uart_online_cfg: model_type=cnn_tcn, ...
```

### 流程 B：当前默认 `CNN-TCN` 先做板端离线对比，再上串口

适用场景：

- 先确认模型在板端本身能跑
- 先排除串口链路问题

步骤：

1. 保持 `paths.kmodel` 和 `paths.scaler_json` 指向 `cnn_tcn` 这套产物
2. 临时把 `runtime.mode` 改成 `csv_cached`
3. 板端执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

如果只想先跑少量样本：

```python
python run_k230_csv_compare.py 10
```

离线结果正常后，再切回：

```json
"runtime": {
  "mode": "uart_online"
}
```

然后再跑在线推理。

### 流程 C：纯 `CNN` 上板在线跑

适用场景：

- 想跑单窗口纯 CNN
- 想和序列模型做对照

当前仓库里对应配置：

- PC 推理配置：
  [infer_config_cnn_20260317.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/infer/infer_config_cnn_20260317.json:1)
- 导出配置：
  [k230_export_config_cnn_20260317.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/export/k230_export_config_cnn_20260317.json:1)

纯 `cnn` 上板时必须满足：

- `model.type = "cnn"`
- `data.sequence_length = 1`

建议板端改成类似这样：

```json
"model": {
  "type": "cnn"
},
"data": {
  "base_window_size": 500,
  "base_step": 200,
  "sequence_length": 1,
  "sequence_step": 1
}
```

然后再把：

- `paths.kmodel`
- `paths.scaler_json`

改成纯 `cnn` 导出的那套文件，例如：

```json
"paths": {
  "kmodel": "model/cnn_all_20260317_030406.kmodel",
  "scaler_json": "model/scaler_20260317_030406.json"
}
```

最后上板执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

启动日志里应看到：

```text
uart_online_cfg: model_type=cnn, ...
```

### 流程 D：旧版 `CNN-LSTM` 上板在线跑

适用场景：

- 要复现旧版 `cnn_lstm_20260320_023445`
- 要和现在的 `cnn_tcn` 做对照

当前仓库里对应配置：

- PC 推理配置：
  [infer_config_cnn_lstm.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/infer/infer_config_cnn_lstm.json:1)
- 导出配置：
  [k230_export_config_cnn_lstm.json](/d:/code/network/canmv_k230_deploy/raw_cnn_pc/configs/export/k230_export_config_cnn_lstm.json:1)

旧版 `cnn_lstm` 上板时建议板端改成：

```json
"model": {
  "type": "cnn_lstm"
},
"data": {
  "base_window_size": 500,
  "base_step": 200,
  "sequence_length": 5,
  "sequence_step": 2
}
```

同时把路径改成旧版这套：

```json
"paths": {
  "kmodel": "model/cnn_lstm_20260320_023445.kmodel",
  "scaler_json": "model/scaler_20260320_023445.json"
}
```

最后板端执行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

启动日志里应看到：

```text
uart_online_cfg: model_type=cnn_lstm, ...
```

### 流程 E：任意模型上板前，先只验证串口链路

适用场景：

- 不确定问题出在模型还是串口
- 想先把协议链路打通

建议顺序：

1. `uart_continuous_send_test.py`
2. `runtime.mode = "uart_echo"`
3. `runtime.mode = "uart_frame_return"`
4. `runtime.mode = "uart_debug_ack"`
5. `runtime.mode = "uart_online"`

其中：

- `uart_echo`：收到什么回什么
- `uart_frame_return`：按大帧/小帧结构回传
- `uart_debug_ack`：每收到 1 帧回 1 个调试 ACK

### 流程 F：导出后只替换板端模型文件，不改逻辑

适用场景：

- 只是换了同结构的新权重
- 不想动脚本

只要模型结构没变，通常只需要改：

- PC 推理侧：
  `weights_path`、`scaler_path`
- 导出侧：
  `weights_pth`、`scaler_pkl`
- 板端侧：
  `paths.kmodel`、`paths.scaler_json`

如果下面任意一项变了，就不能只改路径：

- `base_window_size`
- `base_step`
- `sequence_length`
- `sequence_step`
- `feature_mode`
- `cnn / cnn_lstm / cnn_tcn` 模型结构参数

## 快速选择

如果大将军现在要：

- 跑当前默认模型：
  走“流程 A”
- 先板端验模型，再联调串口：
  走“流程 B”
- 跑纯 CNN：
  走“流程 C”
- 跑旧版 CNN-LSTM：
  走“流程 D”
- 先把串口协议打通：
  走“流程 E”
