# K230 输入、模型、输出配置指南

这份文档专门说明如何修改 K230 runtime 的输入通道、模型绑定和输出槽位。

正式配置入口是：

```text
raw_cnn_k230/configs/runtime.json
```

也可以复制一份新配置，例如：

```text
raw_cnn_k230/configs/runtime_12in12out_model3.json
raw_cnn_k230/configs/runtime_现场名称.json
```

旧的 `k230_config_*.json` 只作为历史兼容文件，不作为当前配置入口。

## 核心概念

runtime 的数据流分三层：

```text
物理输入通道 -> 模型绑定 -> 返回帧输出槽位
```

三层编号都从 0 开始，不是从 1 开始。

例如当前默认 12 路输入、12 路输出时，合法编号都是：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

`input.channel_count` 表示一帧 UART 输入里有多少个物理输入通道。

`output.slot_count` 表示返回帧里有多少个输出槽位。

`models[]` 里每一项表示一个模型。一个模型可以吃一个或多个物理输入通道，也可以和其他模型共用同一个物理输入通道。

## 最重要的规则

每个模型条目里，最关键的是这几项：

```json
{
  "name": "model_1_cnn_tcn",
  "enabled": true,
  "model_type": "cnn_tcn",
  "input_channels": [2, 5],
  "output": {
    "name": "model_1_cnn_tcn",
    "slots": {
      "2": 3,
      "5": 6
    },
    "scale": 100
  },
  "assets": {
    "kmodel": "model/cnn-tcn/xxx.kmodel",
    "scaler_json": "model/cnn-tcn/xxx_scaler.json"
  }
}
```

含义是：

| 字段 | 含义 |
| --- | --- |
| `name` | 模型条目的名字，建议每个模型唯一 |
| `enabled` | 是否启用这个模型 |
| `model_type` | 模型类型，当前常用 `cnn_tcn` |
| `input_channels` | 这个模型要处理哪些物理输入通道 |
| `output.name` | 输出名字前缀，建议和 `name` 一致 |
| `output.slots` 的 key | 物理输入通道编号 |
| `output.slots` 的 value | 该通道的模型结果放到哪个返回帧槽位 |
| `assets.kmodel` | 板端 kmodel 路径，相对 `raw_cnn_k230/` |
| `assets.scaler_json` | scaler JSON 路径，相对 `raw_cnn_k230/` |

最容易混淆的是 `output.slots`：

```json
"slots": {
  "2": 3,
  "5": 6
}
```

这不是“第 2 个输出、第 5 个输出”的意思。

它的意思是：

```text
物理输入通道 2 的模型结果 -> 返回帧 slot 3
物理输入通道 5 的模型结果 -> 返回帧 slot 6
```

## 输出名字怎么生成

如果一个模型的 `output.name` 是：

```text
model_1_cnn_tcn
```

它的 `input_channels` 是：

```json
[2, 5]
```

runtime 会生成两个 named prediction：

```text
model_1_cnn_tcn_ch2
model_1_cnn_tcn_ch5
```

如果 `output.slots` 是：

```json
"slots": {
  "2": 3,
  "5": 6
}
```

最终输出就是：

```text
slot 3 -> model_1_cnn_tcn_ch2
slot 6 -> model_1_cnn_tcn_ch5
```

没有被占用的 slot 返回 `output.default_value`，当前通常是 `0.0`。

## 完整例子：输入 2/3/5/6，两个模型，输出 3/6/8/9

需求：

```text
物理输入通道：2, 3, 5, 6
模型数量：2 个
返回帧输出槽位：3, 6, 8, 9
```

映射关系：

```text
模型 1：
  输入通道 2 -> 输出 slot 3
  输入通道 5 -> 输出 slot 6

模型 2：
  输入通道 3 -> 输出 slot 8
  输入通道 6 -> 输出 slot 9
```

对应的 `models[]` 可以写成：

```json
"models": [
  {
    "name": "model_1_cnn_tcn",
    "enabled": true,
    "model_type": "cnn_tcn",
    "input_channels": [2, 5],
    "output": {
      "name": "model_1_cnn_tcn",
      "slots": {
        "2": 3,
        "5": 6
      },
      "scale": 100
    },
    "assets": {
      "kmodel": "model/cnn-tcn/model_1.kmodel",
      "scaler_json": "model/cnn-tcn/model_1_scaler.json"
    },
    "window": {
      "base_window_size": 500,
      "base_step": 200,
      "sequence_length": 5,
      "sequence_step": 1,
      "feature_mode": "window_demean"
    }
  },
  {
    "name": "model_2_cnn_tcn",
    "enabled": true,
    "model_type": "cnn_tcn",
    "input_channels": [3, 6],
    "output": {
      "name": "model_2_cnn_tcn",
      "slots": {
        "3": 8,
        "6": 9
      },
      "scale": 100
    },
    "assets": {
      "kmodel": "model/cnn-tcn/model_2.kmodel",
      "scaler_json": "model/cnn-tcn/model_2_scaler.json"
    },
    "window": {
      "base_window_size": 500,
      "base_step": 200,
      "sequence_length": 5,
      "sequence_step": 1,
      "feature_mode": "window_demean"
    }
  }
]
```

展开后的最终返回关系是：

| 输出 slot | named prediction | 含义 |
| --- | --- | --- |
| 3 | `model_1_cnn_tcn_ch2` | 模型 1 使用物理输入通道 2 的结果 |
| 6 | `model_1_cnn_tcn_ch5` | 模型 1 使用物理输入通道 5 的结果 |
| 8 | `model_2_cnn_tcn_ch3` | 模型 2 使用物理输入通道 3 的结果 |
| 9 | `model_2_cnn_tcn_ch6` | 模型 2 使用物理输入通道 6 的结果 |
| 其他 slot | 空 | 返回 `output.default_value` |

## 多模型共用输入通道

多个模型可以使用同一个物理输入通道，只要输出 slot 不冲突。

例如两个模型都使用物理输入通道 3：

```json
"models": [
  {
    "name": "model_1_cnn_tcn",
    "input_channels": [3],
    "output": {
      "name": "model_1_cnn_tcn",
      "slots": {
        "3": 7
      }
    }
  },
  {
    "name": "model_2_cnn_tcn",
    "input_channels": [3],
    "output": {
      "name": "model_2_cnn_tcn",
      "slots": {
        "3": 9
      }
    }
  }
]
```

最终关系是：

```text
物理输入通道 3 -> 模型 1 -> slot 7
物理输入通道 3 -> 模型 2 -> slot 9
```

这是允许的，因为 slot 7 和 slot 9 不冲突。

不允许的是两个模型同时写同一个 slot，例如都写 slot 7。校验脚本会报错。

## 添加一个新模型

添加模型时，只改 `models[]`，不用改运行代码。

步骤：

1. 把新模型文件放到 `raw_cnn_k230/model/...` 下。
2. 确认有两个文件：`.kmodel` 和对应的 `_scaler.json`。
3. 在 `models[]` 末尾复制一个已有模型条目。
4. 改 `name` 和 `output.name`，保证模型名清楚且不重复。
5. 改 `input_channels`，写这个模型要吃的物理输入通道。
6. 改 `output.slots`，把每个输入通道映射到唯一输出 slot。
7. 改 `assets.kmodel` 和 `assets.scaler_json`。
8. 运行配置校验。

例子：新增模型 3，吃输入通道 1 和 4，输出到 slot 10 和 11：

```json
{
  "name": "model_3_cnn_tcn",
  "enabled": true,
  "model_type": "cnn_tcn",
  "input_channels": [1, 4],
  "output": {
    "name": "model_3_cnn_tcn",
    "slots": {
      "1": 10,
      "4": 11
    },
    "scale": 100
  },
  "assets": {
    "kmodel": "model/cnn-tcn/model_3.kmodel",
    "scaler_json": "model/cnn-tcn/model_3_scaler.json"
  },
  "window": {
    "base_window_size": 500,
    "base_step": 200,
    "sequence_length": 5,
    "sequence_step": 1,
    "feature_mode": "window_demean"
  }
}
```

## 修改已有模型的输入通道

只改两个地方：

```json
"input_channels": [...]
```

和：

```json
"output": {
  "slots": {
    "...": ...
  }
}
```

例子：原来模型吃输入 0 和 1：

```json
"input_channels": [0, 1],
"output": {
  "slots": {
    "0": 0,
    "1": 1
  }
}
```

现在要改成吃输入 2 和 5，并输出到 slot 3 和 6：

```json
"input_channels": [2, 5],
"output": {
  "slots": {
    "2": 3,
    "5": 6
  }
}
```

注意：`slots` 的 key 必须来自 `input_channels`。如果 `input_channels` 里没有 5，就不要写 `"5": 6`。

## 修改输出槽位

如果只是想换返回帧里的位置，不改模型输入，只改 `output.slots` 的 value。

例如原来：

```json
"input_channels": [2, 5],
"output": {
  "slots": {
    "2": 3,
    "5": 6
  }
}
```

要改成输出到 slot 0 和 1：

```json
"input_channels": [2, 5],
"output": {
  "slots": {
    "2": 0,
    "5": 1
  }
}
```

这表示：

```text
输入通道 2 的结果 -> slot 0
输入通道 5 的结果 -> slot 1
```

## 换模型文件

如果只是换模型版本，不改输入输出映射，只改 `assets`：

```json
"assets": {
  "kmodel": "model/cnn-tcn/new_model.kmodel",
  "scaler_json": "model/cnn-tcn/new_model_scaler.json"
}
```

路径是相对 `raw_cnn_k230/` 的路径。

也就是说实际文件应类似：

```text
raw_cnn_k230/model/cnn-tcn/new_model.kmodel
raw_cnn_k230/model/cnn-tcn/new_model_scaler.json
```

## 禁用或删除模型

临时不用某个模型，可以把：

```json
"enabled": true
```

改成：

```json
"enabled": false
```

禁用后，这个模型不会参与推理，也不会占用输出 slot。

长期不用可以从 `models[]` 里删除整个模型条目。

## 常见合法和非法写法

合法：一个模型吃两个输入，输出到两个不同 slot。

```json
"input_channels": [2, 5],
"output": {
  "slots": {
    "2": 3,
    "5": 6
  }
}
```

合法：两个模型共用同一个输入，但输出到不同 slot。

```text
model_1: 输入 3 -> slot 7
model_2: 输入 3 -> slot 9
```

非法：输出 slot 重复。

```text
model_1: 输入 2 -> slot 3
model_2: 输入 5 -> slot 3
```

非法：输入通道越界。

```text
input.channel_count = 12 时，输入通道 12 是非法的
合法范围只有 0~11
```

非法：`slots` 里写了不属于 `input_channels` 的 key。

```json
"input_channels": [2, 5],
"output": {
  "slots": {
    "3": 8
  }
}
```

这里模型没有吃输入通道 3，所以不应该写 `"3": 8`。

## 异常码规则

异常码跟物理输入通道走，不跟模型走。

例如物理输入通道 3 异常，而模型 1 和模型 2 都使用了通道 3：

```text
model_1_cnn_tcn_ch3 -> slot 7
model_2_cnn_tcn_ch3 -> slot 9
```

那么 slot 7 和 slot 9 会带同一个物理输入通道 3 的异常码。

没有绑定物理输入通道的空 slot，不继承任何输入通道异常码。

## 返回帧规则

当前内层返回帧固定为：

```text
55 AA + 12 * int32(big endian) + FC CF
```

总长度：

```text
52 字节
```

每个输出 slot 对应一个 `int32`。

当前干度值在发送前会乘以 `output.predict_scale`，默认是 100：

```text
0.23 -> 23 -> 00 00 00 17
1.00 -> 100 -> 00 00 00 64
```

`output.value_guard` 限制的是乘以 `predict_scale` 之前的业务值，当前常用范围是：

```text
0.0 ~ 1.0
```

## 修改后的校验流程

改完配置后，先校验配置：

```bash
python scripts/validate_runtime_config.py --config raw_cnn_k230/configs/runtime_你的配置.json
```

重点看输出里的：

```text
runtime config validation ok
models: ...
uart_value_count: 12
legacy_preview.runtime.output.slots
```

`legacy_preview.runtime.output.slots` 就是最终 12 路输出槽位展开结果，应以它为准确认映射。

再生成部署包：

```bash
python scripts/make_deploy_package.py --config raw_cnn_k230/configs/runtime_你的配置.json
```

如果用图形工具，双击：

```text
一键生成部署包.bat
```

然后选择对应的 runtime 配置，点开始生成即可。
