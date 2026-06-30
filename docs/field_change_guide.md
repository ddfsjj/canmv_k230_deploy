# K230 Runtime 协议指南

这份文档说明当前 K230 runtime 的输入、模型、输出和返回帧协议。正式配置入口是：

```text
raw_cnn_k230/configs/runtime.json
```

旧的 `k230_config_*.json` 只作为历史兼容文件，不作为当前协议入口。

## 协议对象

runtime 按下面三层组织数据：

```text
物理输入通道 -> 模型输出 -> 返回帧槽位
```

物理输入通道由 `input.channel_count` 定义总数。当前默认 12 路，编号为：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

返回帧槽位由 `output.slot_count` 定义总数。当前默认 12 路，编号为：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

没有被模型输出占用的槽位返回 `output.default_value`，当前默认值是 `0.0`。

## 模型输出

`models[]` 里的每个启用模型按 `input_channels` 生成模型输出。输出命名规则为：

```text
{模型 output.name}_ch{物理通道编号}
```

例如模型输出名是 `model_1_cnn_tcn`，输入通道是 `0, 1`，则生成：

```text
model_1_cnn_tcn_ch0
model_1_cnn_tcn_ch1
```

## 槽位映射

`models[].output.slots` 定义模型输出进入返回帧的位置。

字段含义：

| 字段 | 含义 |
| --- | --- |
| `slots` 的 key | 物理输入通道编号 |
| `slots` 的 value | 返回帧槽位编号 |

示例：

```json
"input_channels": [0, 1],
"output": {
  "name": "model_1_cnn_tcn",
  "slots": {
    "0": 0,
    "1": 1
  }
}
```

表示：

```text
model_1_cnn_tcn_ch0 -> slot 0
model_1_cnn_tcn_ch1 -> slot 1
```

槽位约束：

- 槽位编号必须满足 `0 <= slot < output.slot_count`。
- 当前 `output.slot_count` 是 12，所以合法槽位是 `0~11`。
- 同一个槽位只能被一个模型输出占用。
- 未占用槽位返回默认值。

## 多模型规则

多个模型可以引用同一个物理输入通道，只要输出槽位不冲突。

例如两个模型都使用物理通道 3：

```text
model_1_cnn_tcn_ch3
model_2_cnn_tcn_ch3
```

可以分别映射到不同槽位：

```text
model_1_cnn_tcn_ch3 -> slot 7
model_2_cnn_tcn_ch3 -> slot 9
```

这两个槽位都来源于物理通道 3，但属于不同模型输出。

## 映射示例

假设协议约定如下：

```text
物理输入通道：0, 1, 3, 5
启用模型：2 个
输出槽位：0, 1, 4, 7, 9
```

并且映射关系为：

| 输出槽位 | named prediction | 含义 |
| --- | --- | --- |
| 0 | `model_1_cnn_tcn_ch0` | 模型 1 使用物理通道 0 的输出 |
| 1 | `model_1_cnn_tcn_ch1` | 模型 1 使用物理通道 1 的输出 |
| 4 | `model_2_cnn_tcn_ch5` | 模型 2 使用物理通道 5 的输出 |
| 7 | `model_1_cnn_tcn_ch3` | 模型 1 使用物理通道 3 的输出 |
| 9 | `model_2_cnn_tcn_ch3` | 模型 2 使用物理通道 3 的输出 |

最终 12 路返回帧槽位为：

```text
slot 0 -> model_1_cnn_tcn_ch0
slot 1 -> model_1_cnn_tcn_ch1
slot 4 -> model_2_cnn_tcn_ch5
slot 7 -> model_1_cnn_tcn_ch3
slot 9 -> model_2_cnn_tcn_ch3
其他 slot -> 默认 0
```

## 异常码规则

异常码跟物理输入通道走，不跟模型走。

例如物理通道 3 触发原始数据异常，而模型 1 和模型 2 都使用了物理通道 3：

```text
model_1_cnn_tcn_ch3 -> slot 7
model_2_cnn_tcn_ch3 -> slot 9
```

那么 slot 7 和 slot 9 会带同一个物理通道 3 异常码。

没有绑定物理输入通道的空槽位，不继承任何物理通道异常码。

## 返回帧规则

当前内层返回帧固定为：

```text
55 AA + 12 * int32(big endian) + FC CF
```

总长度：

```text
52 字节
```

每个输出槽位对应一个 `int32`。当前干度值在进入返回帧前会先按业务值处理，再乘以 `output.predict_scale`。默认规则是：

```text
0.23 -> 23 -> 00 00 00 17
1.00 -> 100 -> 00 00 00 64
```

输出保护范围由 `output.value_guard` 限制。当前默认限制的是乘以 `predict_scale` 之前的业务值范围：

```text
0.0 ~ 1.0
```

## 校验规则

配置校验至少保证：

```text
runtime.json 可以解析
模型文件存在
scaler 文件存在
输入通道不超过 input.channel_count
输出槽位不超过 output.slot_count
输出槽位不重复占用
```

校验命令：

```bash
python scripts/validate_runtime_config.py
```

校验输出里的 `legacy_preview.runtime.output.slots` 是最终 12 路输出槽位展开结果，应以它为准确认协议映射。
