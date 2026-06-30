# UART 返回帧协议

当前统一 runtime 仍沿用旧后端的返回帧格式。

## 返回帧

返回帧是单个内层帧：

```text
55 AA + 12 * int32 + FC CF
```
总长度：

```text
2 + 12 * 4 + 2 = 52 字节
```
当前配置使用：

```text
byte_order = big
value_type = int32
predict_scale = 100
```

## 异常码打包

启用异常码时，每个 int32 的含义是：

```text
[异常码 1 字节][保留 1 字节][干度 uint16]
```

等价计算：

```python
packed = (error_code << 24) | round(prediction * predict_scale)
```

常见异常码：

```text
0x00 正常
0x01 全 0
0x02 低于原始范围
0x03 高于原始范围
0x04 尖峰
0x05 卡死
0x10 满液报警
```
