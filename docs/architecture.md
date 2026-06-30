# K230 统一运行架构

## 目标

板端运行不再区分“单模型脚本”和“多模型脚本”。统一规则是：

```text
输入层：UART/CSV -> 物理通道窗口
推理层：model bindings -> named predictions
输出层：named predictions + channel status -> 12 路返回帧
```

单模型和多模型只由 `raw_cnn_k230/configs/runtime.json` 的 `models[]` 数量决定。

## 配置入口

板端默认只读：

```text
raw_cnn_k230/configs/runtime.json
```

常见修改位置：

- 换模型：改 `models[].assets.kmodel` 和 `models[].assets.scaler_json`
- 改输入通道：改 `models[].input_channels`
- 改输出槽位：改 `models[].output.slot` 或 `models[].output.slots`
- 改异常阈值：改 `status.raw_anomaly`、`status.zero_guard`、`status.full_gas_alarm`
- 改输出保护范围：改 `output.value_guard`
- 改 UART 协议：改 `input.uart` 和 `output.frame`

## 入口兼容

`run_k230_infer.py` 是唯一主入口。

`run_k230_multi_infer.py` 暂时保留为兼容入口，但内部同样转到 `runtime.app`。

当前 `runtime.app` 会把新版 `runtime.json` 转换成统一运行配置，再按模式调度到
`runtime.online` 或 `runtime.csv`。旧入口只负责兼容历史调用方式，不再承载 UART/CSV 主后端。

## 当前抽离进度

- `runtime/config.py`：负责唯一配置校验和旧后端适配。
- `runtime/app.py`：负责根目录探测、配置加载、文件存在性检查和运行模式调度。
- `runtime/online.py`：负责 UART 在线主循环，串起输入、窗口、推理、状态和输出。
- `runtime/csv.py`：负责 CSV 缓存推理，用于离线调试和批量对比。
- `runtime/uart.py`：负责 UART 初始化、52 字节小帧编码、外层大帧/小帧解析。
- `runtime/output.py`：负责输出槽位映射、异常码合成和 UART 返回帧发送。
- `runtime/protocol.py`：负责异常码常量、int32 限幅和异常码/干度打包。
- `runtime/status.py`：负责状态对象创建、raw anomaly 更新、zero guard 入口和 full gas alarm 更新。
- `runtime/bindings.py`：负责模型上下文、输入通道展开、模型输入绑定和 CSV 共享样本推理。
- `runtime/windows.py`：负责在线输入环形缓存、基础窗触发、特征窗口生成和 zero guard 原始序列缓存。
- `runtime/inputs.py`：负责 UART 读取批次、outer/small frame 接收日志和输入批次输出。
