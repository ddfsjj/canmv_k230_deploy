# Anomaly Code Test

这份文档用于板端串口异常码测试。当前返回帧格式：

```text
55 AA + 12 * int32 + FC CF
```

当前配置：

```text
byte_order = big
predict_scale = 100
```

因此单路返回值：

```text
[异常码 1 字节][保留 1 字节][干度 uint16]
```

例子：

```text
00 00 00 17 -> error=0x00, value=0x0017=23 -> 0.23
01 00 00 17 -> error=0x01, value=0x0017=23 -> 0.23
```

## 当前槽位

```text
slot0 = model_1_cnn_tcn_ch0，来源物理通道 0
slot1 = model_1_cnn_tcn_ch1，来源物理通道 1
slot2 = model_2_cnn_tcn_ch0，来源物理通道 0
slot3 = model_2_cnn_tcn_ch1，来源物理通道 1
slot4-slot11 = 空，默认 0
```

异常码跟物理通道走，不跟模型走。

所以：

```text
物理通道 0 异常 -> slot0 和 slot2 同时带异常码
物理通道 1 异常 -> slot1 和 slot3 同时带异常码
```

## 异常码

```text
0x00 正常
0x01 全 0
0x02 低于原始范围
0x03 高于原始范围
0x04 尖峰
0x05 卡死
0x10 满气报警
```

## PC 仿真命令

正常帧：

```bash
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --json
```

物理通道 0 全 0：

```bash
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --raw-errors 0=1 --json
```

期望前 4 路：

```text
slot0: 01 00 00 17
slot1: 00 00 00 1B
slot2: 01 00 00 18
slot3: 00 00 00 15
```

物理通道 1 尖峰：

```bash
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --raw-errors 1=4 --json
```

期望前 4 路：

```text
slot0: 00 00 00 17
slot1: 04 00 00 1B
slot2: 00 00 00 18
slot3: 04 00 00 15
```

物理通道 0 卡死、物理通道 1 高于范围：

```bash
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --raw-errors 0=5,1=3 --json
```

期望前 4 路：

```text
slot0: 05 00 00 17
slot1: 03 00 00 1B
slot2: 05 00 00 18
slot3: 03 00 00 15
```

满气报警仿真：

```bash
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --full-gas --json
```

期望前 4 路：

```text
slot0: 10 00 00 17
slot1: 10 00 00 1B
slot2: 10 00 00 18
slot3: 10 00 00 15
```

## 板端现场测试建议

1. 先确认正常输入时返回：

```text
55 AA 00 00 00 17 00 00 00 1B 00 00 00 18 00 00 00 15 ... FC CF
```

2. 对物理通道 0 输入全 0，期望 slot0/slot2 的首字节变成 `01`。
3. 对物理通道 1 制造尖峰，期望 slot1/slot3 的首字节变成 `04`。
4. 对物理通道 0 输入固定非零值，期望 slot0/slot2 的首字节变成 `05`。
5. 恢复正常输入后，异常码应恢复为 `00`。

如果只看到预测值变化，但高 8 bit 一直是 `00`，优先检查：

```text
runtime.json -> status.raw_anomaly.enabled
runtime.json -> status.raw_anomaly.hit_count / clear_count
runtime.json -> status.raw_anomaly.raw_min / raw_max / spike_max_diff
```
