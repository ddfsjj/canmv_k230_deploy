# K230 Runtime Config

`raw_cnn_k230/configs/runtime.json` 是板端唯一运行配置。板端入口、部署脚本和 PC 仿真脚本都默认读取这一份配置。

## 修改入口

| 目标 | 修改位置 |
| --- | --- |
| 换模型文件 | `models[].assets.kmodel` |
| 换 scaler | `models[].assets.scaler_json` |
| 改模型类型 | `models[].model_type` |
| 改输入物理通道 | `models[].input_channels` |
| 改输出槽位 | `models[].output.slots` |
| 改窗口长度/步长 | `models[].window` |
| 改 UART 参数 | `input.uart` 和 `output.frame` |
| 改异常阈值 | `status.raw_anomaly`、`status.zero_guard`、`status.full_gas_alarm` |
| 改输出保护范围 | `output.value_guard` |
| 单模型/多模型切换 | 只改 `models[]` 数量和 `enabled` |

## 运行结构

```text
UART/CSV
  -> input.channel_count 个物理通道
  -> models[].input_channels 展开为 bindings
  -> named predictions
  -> output.slots 映射到 12 路返回值
  -> 52 字节内层返回帧
```

`binding` 是一个模型和一个输入通道的运行绑定。例如：

```json
{
  "name": "model_1_cnn_tcn",
  "input_channels": [0, 1],
  "output": {
    "name": "model_1_cnn_tcn",
    "slots": {
      "0": 0,
      "1": 1
    }
  }
}
```

会展开为：

```text
model_1_cnn_tcn_ch0 -> slot 0
model_1_cnn_tcn_ch1 -> slot 1
```

如果再加一个模型，仍然只是在 `models[]` 中新增一项，不需要改运行代码。

## 输出槽位

当前默认配置：

| Slot | Named prediction | 来源物理通道 |
| --- | --- | --- |
| 0 | `model_1_cnn_tcn_ch0` | 0 |
| 1 | `model_1_cnn_tcn_ch1` | 1 |
| 2 | `model_2_cnn_tcn_ch0` | 0 |
| 3 | `model_2_cnn_tcn_ch1` | 1 |
| 4-11 | 空 | 默认 0 |

异常码跟物理通道走，不跟模型走。所以物理通道 0 异常时，slot 0 和 slot 2 都会带同一个异常码。

## 输出保护

`output.value_guard` 用于限制最终进入 12 路返回帧的预测值，避免模型偶发异常值直接返回到上位机。

当前默认：

```json
{
  "enabled": true,
  "min": 0.0,
  "max": 1.0,
  "replace_non_finite_with": 0.0
}
```

这里限制的是乘以 `predict_scale` 之前的业务值。当前干度按 `0.23 -> 23` 返回，
所以默认范围是 `0.0~1.0`。

## 常用命令

校验当前默认配置：

```bash
python scripts/validate_runtime_config.py
```

校验单模型示例：

```bash
python scripts/validate_runtime_config.py --config raw_cnn_k230/configs/runtime.single.example.json
```

校验多模型示例：

```bash
python scripts/validate_runtime_config.py --config raw_cnn_k230/configs/runtime.multi.example.json
```

仿真输出帧：

```bash
python scripts/run_runtime_sim.py
```

生成部署包：

```bash
python scripts/make_deploy_package.py --clean
```

## 示例配置

`raw_cnn_k230/configs/runtime.single.example.json` 表示：

```text
1 个模型
输入 ch0/ch1
输出 slot 0/1
```

`raw_cnn_k230/configs/runtime.multi.example.json` 表示：

```text
2 个模型
每个模型都输入 ch0/ch1
输出 slot 0/1/2/3
```

正式部署只读取 `raw_cnn_k230/configs/runtime.json`。示例文件用于复制参考，不会被默认部署脚本打包。
