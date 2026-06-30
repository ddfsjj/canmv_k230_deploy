# PC Model Definition Audit

## 当前更新

状态：重复模型定义清理完成。`infer.py` 和 `build_kmodel.py` 不再维护本地 `CNNAll / CNNLSTM / CNNTCN / CNNTCNSeg3SoftStatsMoE` 类，模型结构统一由 `raw_cnn_pc/raw_cnn/models.py` 承载。

`build_model_from_config()` 已收口到公共层；两个脚本里的同名函数只保留为兼容转调入口。`build_kmodel.py` 仍保留 `Seg3ExportWrapper`，因为它属于 ONNX/KModel 导出包装流程，不是模型结构定义。

状态：`CNNAll`、`CNNTCNSeg3SoftStatsMoE` 已经迁移到 `raw_cnn_pc/raw_cnn/models.py`。

当前公共模型层已经承载：

```text
CNNAll
CNNLSTM
CausalConv1d
TemporalBlock
CNNTCN
CNNTCNSeg3SoftStatsMoE
normalize_model_type
infer_lstm_layout_from_state_dict
```

`CNNAll` 统一采用 `build_kmodel.py` 侧更稳的实现：当 `pool_size <= 1` 时使用 `Identity()`，避免 `MaxPool1d(1)` 或异常配置在导出链路里产生额外风险。`infer.py` 和 `build_kmodel.py` 当前都会通过 `raw_cnn.models.build_model_from_config()` 构建模型。

旧脚本里的本地 `CNNAll / CNNLSTM / CNNTCN / CNNTCNSeg3SoftStatsMoE` 类已经清理，后续模型结构只维护 `raw_cnn_pc/raw_cnn/models.py`。

`Seg3` 公共实现采用：

```text
forward()       -> prediction tensor
forward_debug() -> dict，包含 route_logits / route_probs / expert_preds / head_input
```

这样 `build_kmodel.py` 的 ONNX/KModel 导出仍保持 tensor-only 输出，`infer.py` 和对比脚本需要调试信息时可以继续通过 `forward_debug()` 获取。

剩余整理项：

```text
Seg3ExportWrapper 仍保留在 build_kmodel.py，属于导出包装器，不是模型结构定义
```

这份审计记录 `raw_cnn_pc/infer.py` 和 `raw_cnn_pc/build_kmodel.py` 中模型定义的差异，用于决定后续把模型结构迁移到 `raw_cnn_pc/raw_cnn/models.py` 的顺序。

## 结论

当前可以直接统一的部分：

```text
ensure_per_layer
CNNLSTM
CausalConv1d
TemporalBlock
CNNTCN
normalize_model_type
infer_lstm_layout_from_state_dict
```

需要谨慎处理的部分：

```text
CNNAll
CNNTCNSeg3SoftStatsMoE
Seg3ExportWrapper
build_model_from_config
```

## 差异表

| 对象 | 状态 | 说明 |
| --- | --- | --- |
| `ensure_per_layer` | 一致 | 两边源码 hash 一致 |
| `CNNLSTM` | 一致 | 两边源码 hash 一致，可优先迁移 |
| `CausalConv1d` | 一致 | 两边源码 hash 一致，可优先迁移 |
| `TemporalBlock` | 一致 | 两边源码 hash 一致，可优先迁移 |
| `CNNTCN` | 一致 | 两边源码 hash 一致，可优先迁移 |
| `normalize_model_type` | 一致 | 两边源码 hash 一致 |
| `infer_lstm_layout_from_state_dict` | 一致 | 两边源码 hash 一致 |
| `CNNAll` | 不一致 | `build_kmodel.py` 支持 `pool_size <= 1` 时用 `Identity`，`infer.py` 直接用 `MaxPool1d` |
| `CNNTCNSeg3SoftStatsMoE` | 不一致 | `infer.py` 返回调试字典，`build_kmodel.py` 直接返回 prediction tensor |
| `Seg3ExportWrapper` | 只在导出侧存在 | 用于导出双输入 ONNX：`input` 和 `raw_input` |
| `build_model_from_config` | 基本一致 | 实际逻辑一致，只有 LSTM 注释文本不同 |

## 关键差异

### CNNAll

`infer.py`:

```python
self.pools = nn.ModuleList([nn.MaxPool1d(int(p)) for p in pool_sizes])
```

`build_kmodel.py`:

```python
self.pool_sizes = [int(p) for p in pool_sizes]
self.pools = nn.ModuleList(
    [nn.MaxPool1d(p) if p > 1 else nn.Identity() for p in self.pool_sizes]
)
```

影响：

```text
当 pool_size 全部 > 1 时，两边行为一致。
当 pool_size 为 1 或更小配置被误传时，导出侧更稳，推理侧可能不兼容。
```

建议：

```text
统一时采用 build_kmodel.py 的 CNNAll 版本。
```

### Seg3

`infer.py` 的 `CNNTCNSeg3SoftStatsMoE.forward()` 返回：

```text
{
  prediction,
  route_logits,
  route_probs,
  expert_preds,
  low_pred,
  mid_pred,
  high_pred,
  head_input
}
```

`build_kmodel.py` 的 `CNNTCNSeg3SoftStatsMoE.forward()` 返回：

```text
prediction tensor
```

导出侧额外有：

```text
Seg3ExportWrapper
```

用于把双输入导出为 ONNX：

```text
input
raw_input
```

影响：

```text
Seg3 不能直接粗暴迁移成单一 forward 返回值。
推理侧需要调试信息，导出侧需要 tensor-only 输出。
```

建议：

```text
统一模型类时，让 CNNTCNSeg3SoftStatsMoE.forward() 保持返回 prediction tensor。
如果 PC 推理需要调试信息，新增 forward_debug() 或 return_debug 参数。
导出继续保留 Seg3ExportWrapper。
```

## 推荐迁移顺序

### 第一步：迁移无差异模型

状态：已完成第一阶段迁移。

先把以下内容移入 `raw_cnn_pc/raw_cnn/models.py`：

```text
ensure_per_layer
CNNLSTM
CausalConv1d
TemporalBlock
CNNTCN
normalize_model_type
infer_lstm_layout_from_state_dict
```

然后让：

```text
infer.py
build_kmodel.py
```

都从 `raw_cnn.models` 导入这些定义。

当前实现：

```text
raw_cnn_pc/raw_cnn/models.py
  - 承载 CNNLSTM / CNNTCN / CausalConv1d / TemporalBlock
  - 承载 normalize_model_type / infer_lstm_layout_from_state_dict
  - 提供 build_shared_model_from_config()

raw_cnn_pc/infer.py
  - CNN-TCN / CNN-LSTM 构建走 raw_cnn.models
  - CNNAll / Seg3 已迁移到 raw_cnn.models

raw_cnn_pc/build_kmodel.py
  - CNN-TCN / CNN-LSTM 构建走 raw_cnn.models
  - CNNAll / Seg3 构建走 raw_cnn.models
  - Seg3ExportWrapper 保留为导出包装器
```

### 第二步：迁移 CNNAll

采用 `build_kmodel.py` 的实现作为统一版本，因为它兼容 `pool_size <= 1`：

```text
MaxPool1d(p) if p > 1 else Identity()
```

### 第三步：处理 Seg3

建议设计成：

```text
CNNTCNSeg3SoftStatsMoE.forward()       -> prediction tensor
CNNTCNSeg3SoftStatsMoE.forward_debug() -> dict
Seg3ExportWrapper                      -> ONNX 双输入导出包装
```

然后改 `infer.py` 中 Seg3 推理分支：

```text
outputs = model.forward_debug(...)
y_pred = model.compose_prediction(outputs["prediction"])
```

或者如果不需要调试信息：

```text
prediction = model(...)
y_pred = model.compose_prediction(prediction)
```

### 第四步：统一 build_model_from_config

`build_model_from_config` 逻辑目前基本一致，可以在模型类迁移完成后统一迁入 `raw_cnn_pc/raw_cnn/models.py`。

## 当前风险判断

短期不建议马上大改 Seg3。更稳的下一步是：

```text
1. 迁移 CNNLSTM/CNNTCN 等完全一致的定义
2. 跑 AST 检查
3. 在有 torch 的环境跑 infer.py quick smoke
4. 再迁移 CNNAll
5. 最后单独处理 Seg3
```
