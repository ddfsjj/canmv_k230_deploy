# Raw CNN（PC + K230）完整流程说明

本文档基于当前仓库里的这两部分代码：
- `raw_cnn_pc/`：PC 侧验证 `.pth`、导出 `.onnx/.kmodel`
- `raw_cnn_k230/`：K230 板端运行 `.kmodel`

本文档重点解决 4 个实际问题：
- 换了新的 `pth` 之后，PC 侧到底先改哪里
- PC 侧要先后运行哪些脚本
- K230 侧哪些地方必须改，哪些地方不用改
- 样本数量如何写到配置里，而不是每次都在命令行输入

## 1. 目录分工

`raw_cnn_pc/` 负责：
- 使用 `infer.py` 在 PC 上验证 `.pth` 推理结果
- 使用 `build_kmodel.py` 导出 `onnx`、`scaler json`、`calibration npy`，并编译生成 `kmodel`
- 按 `k230_export_config.json` 中的目标路径，把导出产物写到 `../raw_cnn_k230/model/`

`raw_cnn_k230/` 负责：
- 使用 `run_k230_infer.py` 在板端加载 `kmodel` 和 `scaler json`
- 读取 `k230_config.json` 中的路径、切窗参数、运行模式、UART 参数
- 离线模式下读取本地 `test_data/*.csv`
- 在线模式下通过 UART 实时收数据并返回结果
- 使用 `run_k230_csv_compare.py` 做板端离线对比

## 2. 三个关键配置文件分别控制什么

### 2.1 `raw_cnn_pc/infer_config.json`

它决定 PC 上 `infer.py` 怎么跑：
- `data.test_data_dir`：PC 侧测试数据目录
- `data.base_window_size/base_step/sequence_length/sequence_step`：切窗参数
- `preprocessing.feature_mode`：当前预处理方式，当前常用是 `window_demean`
- `model.conv_filters/kernel_size/pool_size`：网络结构参数
- `model.weights_path`：要加载的 `.pth`
- `normalization.scaler_path`：要加载的 `.pkl`
- `runtime.max_samples`：PC 侧默认跑多少条样本，`null` 表示全量

一句话：这个文件控制“PC 用什么模型、什么切窗参数、什么 scaler、默认跑多少条样本去做 `.pth` 推理”。

### 2.2 `raw_cnn_pc/k230_export_config.json`

它决定 `build_kmodel.py` 导出什么、导出到哪里：
- `paths.weights_pth`：导出时使用的 `.pth`
- `paths.scaler_pkl`：导出时使用的 `.pkl`
- `paths.onnx`：导出的 `.onnx` 路径
- `paths.kmodel`：导出的 `.kmodel` 路径
- `paths.scaler_json`：导出的 `scaler json` 路径
- `paths.calibration_npy`：量化校准样本保存路径
- `data.*`：切窗参数，必须和训练时一致
- `preprocessing.feature_mode`：导出时的预处理方式，必须和 PC / K230 一致
- `model.*`：网络结构参数，必须和训练时一致
- `quantization.*`：量化参数

一句话：这个文件控制“把哪个 `.pth` 编译成哪个 `.kmodel`，并把产物放到 `raw_cnn_k230/` 的哪里”。

当前仓库里，这个配置文件已经额外带了一个 `_help` 说明区块。
大将军以后如果忘了某个字段是什么意思，直接打开 `k230_export_config.json` 看 `_help` 即可，不用先翻代码。

量化校准这块以后最常改的就是这几个字段：
- `paths.calibration_data_dir`：量化校准数据目录
- `quantization.samples_count`：量化校准样本条数
- `quantization.sampling_strategy`：量化校准样本抽取方式
- `quantization.random_seed`：随机抽样时的固定种子
- `quantization.calibrate_method`：量化范围估计方法

如果你只是以后换一批量化数据，通常只需要改：
- `paths.calibration_data_dir`

### 2.3 `raw_cnn_k230/k230_config.json`

它决定 K230 板端运行时读哪些文件、按什么模式跑：
- `paths.kmodel`：板端加载哪个 `.kmodel`
- `paths.scaler_json`：板端加载哪个 `scaler json`
- `paths.test_data_dir`：离线模式下测试数据目录
- `paths.predictions_csv`：板端输出的预测 CSV 文件名
- `data.*`：板端切窗参数，必须和导出侧一致
- `preprocessing.feature_mode`：板端预处理方式，必须和 PC / 导出侧一致
- `runtime.mode`：运行模式，例如 `uart_online` 或 `csv_cached`
- `runtime.max_samples/infer_batch_size/write_predictions_csv`：`run_k230_infer.py` 离线模式时控制样本数和输出
- `runtime.compare_max_samples`：`run_k230_csv_compare.py` 默认跑多少条，`null` 表示全量
- `uart.*`：串口参数

一句话：这个文件控制“K230 实际运行时读哪个模型、用哪套参数、默认跑多少条、以哪种模式运行”。

## 3. 样本数量现在优先写在配置里

当前代码已经支持“直接点运行脚本，就按配置文件里的样本数量执行”。

### 3.1 PC 侧怎么控制数量

看 `raw_cnn_pc/infer_config.json`：

```json
"runtime": {
  "device": "cpu",
  "max_samples": null
}
```

含义如下：
- `max_samples: null`：全量跑
- `max_samples: 10`：只跑前 10 条
- `max_samples: 50`：只跑前 50 条

也就是说：
- 你直接点运行 `infer.py`，它默认就读这里
- 你不想每次命令行传 `--max_samples`，就改这个字段

### 3.2 K230 离线对比怎么控制数量

看 `raw_cnn_k230/k230_config.json`：

```json
"runtime": {
  "compare_max_samples": 10
}
```

含义如下：
- `compare_max_samples: null`：全量跑
- `compare_max_samples: 10`：只跑前 10 条
- `compare_max_samples: 50`：只跑前 50 条

也就是说：
- 你直接点运行 `run_k230_csv_compare.py`，它默认就读这里
- 你不想每次改脚本或命令行，就改这个字段

### 3.3 什么时候还需要命令行参数

命令行参数现在仍然能用，但建议只在“临时覆盖配置”的时候使用。

例如：
- 平时默认跑全量，配置里写 `null`
- 今天只想临时抽 10 条，就命令行传 `--max_samples 10`

如果没有这种临时需求，建议一直改配置，不要每次都在输入行里写数量。

## 4. 第一次完整跑通流程

下面默认命令都在仓库根目录执行：`d:\code\network\canmv_k230_deploy`

### 4.1 安装依赖

先安装 PC 推理依赖：

```bash
cd raw_cnn_pc
pip install -r requirements.txt
```

如果还需要导出 `kmodel`，再安装导出依赖：

```bash
cd raw_cnn_pc
pip install -r requirements_k230_host.txt
```

如果 `nncase` 报 `.NET Runtime` 相关错误，再安装 .NET Runtime 7：

```bash
winget install --id Microsoft.DotNet.Runtime.7 -e --silent --accept-package-agreements --accept-source-agreements
```

### 4.2 先在 PC 验证 `.pth`

如果你已经在 `infer_config.json` 里写好了 `runtime.max_samples`，直接运行：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json
```

如果你想临时覆盖配置，也可以这样运行：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --max_samples 10 --output predictions_pc_quick.csv
```

确认以下几项：
- 没有 `load_state_dict` 报错
- 日志里能看到 `samples`、`input_shape`、`MAE`、`RMSE`
- 生成了预测 CSV

如果这里只跑不通，不要继续导出 `kmodel`，先把 PC 侧模型配置问题解决。

### 4.3 导出 `kmodel`

完整导出并编译：

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json
```

只想先导出 `onnx/scaler/calibration`，先不编译：

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json --skip_compile
```

需要临时改校准样本数：

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json --max_calib_samples 128
```

### 4.4 检查导出产物

按当前仓库默认配置，导出产物会写到：
- `raw_cnn_k230/model/cnn_all_20260317_030406.onnx`
- `raw_cnn_k230/model/cnn_all_20260317_030406.kmodel`
- `raw_cnn_k230/model/scaler_20260317_030406.json`
- `raw_cnn_k230/model/calibration_input.npy`

这里一定要看一眼：
- `raw_cnn_k230/model/` 下是否确实生成了新的 `.onnx/.kmodel/.json/.npy`
- 文件名是否和你期望部署到板端的文件名一致

### 4.5 拷贝到板端

建议直接拷贝整个目录：
- 本地 `raw_cnn_k230/`
- 板端 `/sdcard/raw_cnn_k230/`

至少需要这些文件：
- `run_k230_infer.py`
- `run_k230_csv_compare.py`
- `k230_config.json`
- `model/*.kmodel`
- `model/*.json`
- `test_data/*.csv`，如果你要做板端离线对比

### 4.6 板端运行

如果你要跑正常在线 UART 推理：

```python
cd /sdcard/raw_cnn_k230
python run_k230_infer.py
```

如果你要跑板端离线对比：

先看 `raw_cnn_k230/k230_config.json` 里的：
- `runtime.compare_max_samples`

然后直接运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

这时：
- `compare_max_samples = 10`：只跑前 10 条
- `compare_max_samples = null`：全量跑

## 5. 换了新的 `pth` 之后，PC 侧到底要怎么操作

这一节最常用，建议严格按顺序做。

### 5.1 情况 A：只换了权重和 scaler，网络结构没变

适用前提：
- 还是当前这套 `CNN-All`
- `conv_filters/kernel_size/pool_size` 不变
- `base_window_size/base_step/sequence_length/sequence_step` 不变
- `feature_mode` 不变

这时你只需要改“文件路径”和“默认跑多少条”，不需要改网络结构参数。

#### 第 1 步：把新文件放到 PC 模型目录

把新的：
- `xxx.pth`
- `xxx.pkl`

放到：
- `raw_cnn_pc/model/`

建议文件名带日期或版本号，避免覆盖不清楚，例如：
- `raw_cnn_pc/model/cnn_all_20260327_xxx.pth`
- `raw_cnn_pc/model/scaler_20260327_xxx.pkl`

#### 第 2 步：修改 `infer_config.json`

至少改这两个字段：
- `model.weights_path`
- `normalization.scaler_path`

例如：

```json
"model": {
  "weights_path": "model/cnn_all_20260327_xxx.pth"
},
"normalization": {
  "scaler_path": "model/scaler_20260327_xxx.pkl"
}
```

如果你还想控制“直接点运行 `infer.py` 时默认跑多少条”，同时设置：
- `runtime.max_samples`

建议：
- 联调时设成 `10` 或 `20`
- 正式全量验证时设成 `null`

#### 第 3 步：修改 `k230_export_config.json`

至少改这两个字段：
- `paths.weights_pth`
- `paths.scaler_pkl`

如果你希望导出的板端文件名也换版本，建议同时改：
- `paths.onnx`
- `paths.kmodel`
- `paths.scaler_json`

例如：

```json
"paths": {
  "weights_pth": "model/cnn_all_20260327_xxx.pth",
  "scaler_pkl": "model/scaler_20260327_xxx.pkl",
  "onnx": "../raw_cnn_k230/model/cnn_all_20260327_xxx.onnx",
  "kmodel": "../raw_cnn_k230/model/cnn_all_20260327_xxx.kmodel",
  "scaler_json": "../raw_cnn_k230/model/scaler_20260327_xxx.json"
}
```

#### 第 4 步：先跑 PC 侧 `.pth` 验证

如果你已经在 `infer_config.json` 里设好了数量，直接运行：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json
```

如果你想临时覆盖配置，再使用命令行参数。

例如快速冒烟：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --max_samples 10 --output predictions_pc_quick.csv
```

例如全量：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --output predictions_pc_all.csv
```

先看 4 件事：
- 能否正常加载新 `.pth`
- `input_shape` 是否符合预期
- `MAE/RMSE` 是否正常
- 预测 CSV 是否已生成

#### 第 5 步：PC 验证通过后，再导出 `kmodel`

```bash
cd raw_cnn_pc
python build_kmodel.py --config k230_export_config.json
```

导出完成后，检查 `raw_cnn_k230/model/` 里是否出现了你刚才配置的新文件名。

### 5.2 情况 B：不仅换了权重，连网络结构或切窗参数也变了

如果下面任意一项改了，就不是“只换路径”，而是“配置联动修改”：
- `conv_filters`
- `kernel_size`
- `pool_size`
- `base_window_size`
- `base_step`
- `sequence_length`
- `sequence_step`
- `feature_mode`

这时必须同步修改 3 个地方。

#### PC 推理配置要改

文件：`raw_cnn_pc/infer_config.json`

需要同步：
- `data.*`
- `preprocessing.feature_mode`
- `model.conv_filters/kernel_size/pool_size`
- `model.weights_path`
- `normalization.scaler_path`
- `runtime.max_samples`，按你的联调需要设置

#### K230 导出配置要改

文件：`raw_cnn_pc/k230_export_config.json`

需要同步：
- `data.*`
- `preprocessing.feature_mode`
- `model.conv_filters/kernel_size/pool_size`
- `paths.weights_pth`
- `paths.scaler_pkl`
- 必要时 `paths.onnx/kmodel/scaler_json`

#### K230 运行配置要改

文件：`raw_cnn_k230/k230_config.json`

至少同步：
- `data.*`
- `preprocessing.feature_mode`

如果你导出的文件名也变了，还要同步：
- `paths.kmodel`
- `paths.scaler_json`

如果你要做板端离线对比，还顺便看一下：
- `runtime.compare_max_samples`

否则会出现这些典型问题：
- `infer.py` 侧 `load_state_dict(strict=True)` 报 shape 或 key 不匹配
- `build_kmodel.py` 导出出来的模型输入尺寸和板端配置不一致
- K230 侧切窗参数不一致，导致输入内容不一样，预测值明显漂移

## 6. 换新模型后，K230 文件夹里到底要改什么

这个问题要分情况。

### 6.1 如果你只是换了 `pth/pkl`，但导出后的板端文件名没变

例如你仍然导出成：
- `raw_cnn_k230/model/cnn_all_20260317_030406.kmodel`
- `raw_cnn_k230/model/scaler_20260317_030406.json`

那 K230 侧通常只需要：
- 确认 `raw_cnn_k230/model/` 下的文件已经被新的导出结果覆盖
- 不需要改 `k230_config.json` 里的 `paths.kmodel/scaler_json`

也就是说，K230 侧只拷贝新文件即可。

### 6.2 如果你导出的 `kmodel/json` 文件名变了

例如你新导出成：
- `raw_cnn_k230/model/cnn_all_20260327_xxx.kmodel`
- `raw_cnn_k230/model/scaler_20260327_xxx.json`

那 K230 侧必须改 `raw_cnn_k230/k230_config.json`：

```json
"paths": {
  "kmodel": "model/cnn_all_20260327_xxx.kmodel",
  "scaler_json": "model/scaler_20260327_xxx.json"
}
```

否则板端仍然会去读旧文件。

### 6.3 如果切窗参数或预处理变了

还必须改 `raw_cnn_k230/k230_config.json` 里的：
- `data.base_window_size`
- `data.base_step`
- `data.sequence_length`
- `data.sequence_step`
- `preprocessing.feature_mode`

注意：这里必须和 `infer_config.json`、`k230_export_config.json` 保持一致，不能只改一边。

### 6.4 如果你只关心板端离线对比跑多少条

只改这一个字段即可：

```json
"runtime": {
  "compare_max_samples": 10
}
```

需要全量就写：

```json
"runtime": {
  "compare_max_samples": null
}
```

## 7. 推荐的实际执行顺序

每次换模型，按这个顺序做最稳。

### 7.1 PC 侧顺序

1. 把新的 `pth/pkl` 放到 `raw_cnn_pc/model/`
2. 改 `raw_cnn_pc/infer_config.json` 的 `weights_path/scaler_path`
3. 按需要设置 `raw_cnn_pc/infer_config.json` 的 `runtime.max_samples`
4. 改 `raw_cnn_pc/k230_export_config.json`
5. 如果结构或切窗变了，同时准备修改 `raw_cnn_k230/k230_config.json`
6. 跑 `infer.py` 先验证 `.pth`
7. 验证没问题后，跑 `build_kmodel.py`
8. 检查 `raw_cnn_k230/model/` 是否生成了新的 `.kmodel/.json`

### 7.2 K230 侧顺序

1. 如果导出文件名变了，先改 `raw_cnn_k230/k230_config.json` 的 `paths.kmodel/scaler_json`
2. 如果切窗或预处理变了，改 `raw_cnn_k230/k230_config.json` 的 `data.*` 和 `preprocessing.feature_mode`
3. 如果要做本地 CSV 对比，确认 `runtime.compare_max_samples` 是你要的数量
4. 把更新后的 `raw_cnn_k230/` 整体拷到板端
5. 如果要做本地 CSV 对比，运行 `python run_k230_csv_compare.py`
6. 如果要做串口在线推理，运行 `python run_k230_infer.py`

## 8. PC 与 K230 结果对比怎么做

### 8.1 PC 侧先生成对比 CSV

先在 `raw_cnn_pc/infer_config.json` 里设置：

```json
"runtime": {
  "device": "cpu",
  "max_samples": 10
}
```

然后直接运行：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --output predictions_pc_compare.csv
```

生成：
- `raw_cnn_pc/predictions_pc_compare.csv`

如果想全量对比，把 `runtime.max_samples` 改成 `null`，再运行同一条命令即可。

### 8.2 板端生成对比 CSV

先在 `raw_cnn_k230/k230_config.json` 里设置：

```json
"runtime": {
  "compare_max_samples": 10
}
```

然后直接运行：

```python
cd /sdcard/raw_cnn_k230
python run_k230_csv_compare.py
```

生成：
- `/sdcard/raw_cnn_k230/predictions_k230_compare.csv`

`run_k230_csv_compare.py` 会自动做这些事：
- 强制走 `csv_cached`
- 固定从第 0 条样本开始
- 默认读取 `runtime.compare_max_samples`
- 自动关闭 UART
- 自动写 `predictions_k230_compare.csv`

如果你想全量对比，就把：

```json
"compare_max_samples": null
```

写进 `k230_config.json`，然后重新运行同一个脚本。

### 8.3 两边如何对齐

要让 PC 和 K230 对比有意义，至少保证这几项一致：
- 两边都从第 0 条样本开始
- 两边数量一致，例如都跑 10 条或都跑全量
- 两边使用的是同一份测试数据
- `data.*` 和 `preprocessing.feature_mode` 完全一致

## 9. 数据与输入约束

- 每个 CSV 只读取第 1 列
- 非数值会被跳过
- 标签从文件名提取，规则是取 `-` 前的前缀并转成 `float`
- 例如 `0.2097-18.79-20.csv` 的标签会被解析成 `0.2097`
- 文件名如果不符合这个规则，标签会变成 `NaN`

## 10. 常见问题排查

### 10.1 `load_state_dict` 报错

通常说明：
- `.pth` 和 `infer_config.json` 里的 `model.*` 不匹配
- 结构变了，但你只改了路径，没有改 `conv_filters/kernel_size/pool_size`

### 10.2 `No valid samples found under ...`

通常说明：
- `test_data_dir` 路径不对
- CSV 太短，不满足 `base_window_size`
- `sequence_length` 太大，切不出样本

### 10.3 `ONNX export requires onnx package`

说明导出依赖没装：

```bash
pip install -r requirements_k230_host.txt
```

### 10.4 `nncase is not installed` 或编译失败

通常说明：
- 当前 Python 环境不是你装 `nncase` 的那个环境
- .NET Runtime 未安装完整

### 10.5 板端能跑，但结果明显不对

优先检查这 4 项是否三边一致：
- `data.*`
- `preprocessing.feature_mode`
- `paths.kmodel/scaler_json`
- PC 与板端是否用的是同一份测试数据

### 10.6 板端 UART 没数据

检查：
- `uart.enabled`
- `uart_id`
- `tx_pin/rx_pin`
- `baudrate`
- 上位机帧格式是否和 `header/tail/value_type/byte_order/value_count` 一致

### 10.7 明明直接点运行，为什么数量不对

先检查：
- PC 侧：`raw_cnn_pc/infer_config.json` 的 `runtime.max_samples`
- K230 侧：`raw_cnn_k230/k230_config.json` 的 `runtime.compare_max_samples`

很多时候不是脚本有问题，而是配置文件里的值还停留在之前联调用的数量。

## 11. 最短执行版

如果你只是换了一个新的 `.pth/.pkl`，并且网络结构没变，最短流程就是：

1. 把新 `.pth/.pkl` 放进 `raw_cnn_pc/model/`
2. 改 `raw_cnn_pc/infer_config.json` 里的 `weights_path/scaler_path`
3. 按需要改 `raw_cnn_pc/infer_config.json` 里的 `runtime.max_samples`
4. 改 `raw_cnn_pc/k230_export_config.json` 里的 `weights_pth/scaler_pkl`
5. 按需要改 `raw_cnn_k230/k230_config.json` 里的 `runtime.compare_max_samples`
6. 跑：

```bash
cd raw_cnn_pc
python infer.py --config infer_config.json --output predictions_pc_quick.csv
python build_kmodel.py --config k230_export_config.json
```

7. 如果导出的 `kmodel/json` 文件名变了，再改 `raw_cnn_k230/k230_config.json` 的 `paths.kmodel/scaler_json`
8. 把 `raw_cnn_k230/` 拷到板端
9. 板端跑：

```python
python run_k230_csv_compare.py
```

或：

```python
python run_k230_infer.py
```

照这个顺序做，一般不会乱。
