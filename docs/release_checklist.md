# Release Checklist

这份清单用于每次生成板端部署包前后检查，避免配置、模型文件、返回协议或上位机解析错位。

## 1. 配置校验

```bash
python scripts/validate_runtime_config.py
```

必须确认：

```text
runtime config validation ok
profile 是本次要部署的 profile
models 数量符合预期
uart_value_count = 12
```

## 2. PC 输出帧仿真

默认仿真：

```bash
python scripts/run_runtime_sim.py
```

使用现场推理值仿真：

```bash
python scripts/run_runtime_sim.py --values 0.226196,0.265625,0.237549,0.205078
```

异常码仿真：

```bash
python scripts/run_runtime_sim.py --values 0.226196,0.265625,0.237549,0.205078 --raw-errors 0=1
```

更多异常码串口对照见 [anomaly_test.md](anomaly_test.md)。

必须确认：

```text
frame_len = 52
slot 0/1/2/3 对应预期 named prediction
slot 4-11 为 0
异常码在 packed_int32 高 8 bit
```

## 3. 生成部署包

```bash
python scripts/make_deploy_package.py --clean
```

必须确认生成：

```text
deploy_pkg/boot.py
deploy_pkg/main.py
deploy_pkg/raw_cnn_k230/configs/runtime.json
deploy_pkg/raw_cnn_k230/runtime/
deploy_pkg/raw_cnn_k230/model/...
deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

## 4. 检查 Manifest

先运行：

```bash
python scripts/verify_deploy_package.py
```

必须确认：

```text
deploy package verification ok
```

打开：

```text
deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

必须确认：

```text
profile_name 正确
config.sha256 有值
root_file_records 包含 boot.py/main.py
assets 包含所有 kmodel 和 scaler_json
每个 asset 都有 bytes 和 sha256
```

## 5. 拷贝到 SD 卡

部署方式固定为：

```text
deploy_pkg/* -> /sdcard/
```

不要只拷贝 `deploy_pkg/raw_cnn_k230/`，否则 SD 根目录 `main.py` 不会更新。

## 6. 板端启动检查

软重启后 IDE 应看到：

```text
SD root launcher: raw_cnn_k230 unified runtime
runtime_config: /sdcard/raw_cnn_k230/configs/runtime.json
=== K230 Unified Runtime ===
config_path: /sdcard/raw_cnn_k230/configs/runtime.json
mode: uart_online
model_bindings: ...
UART sender enabled: UART2, 921600 bps, ...
```

如果缺文件，应看到明确路径，例如：

```text
models[0] model_1_cnn_tcn kmodel missing: /sdcard/raw_cnn_k230/model/...
```

## 7. 首次推理检查

正常输入后应看到：

```text
uart_online_trigger: infer_round_next=1, rx_small_frame_idx=1300, ...
uart_online_result: infer_round=1, preds=[...], raw_error_codes=[...]
```

当前配置下，首次推理在 `1300` 小帧左右触发是预期行为。

## 8. 上位机返回帧检查

单帧应为 52 字节：

```text
55 AA
12 * int32
FC CF
```

当前 int32 协议：

```text
[异常码 1 字节][保留 1 字节][干度 uint16]
```

示例：

```text
00 00 00 17 -> 异常码 0x00，干度 23，实际值 0.23
01 00 00 17 -> 异常码 0x01，干度 23，实际值 0.23
```

必须确认：

```text
slot 0/1/2/3 与板端 preds 四舍五入后一致
slot 4-11 为 0
异常码解析正确
连续两帧时按 52 字节切分
```

## 9. 日志级别

上线默认建议：

```json
"debug_outer_rx": false,
"debug_predict_trace": true
```

稳定运行后可以考虑：

```json
"debug_predict_trace": false
```

调试 UART 接收节拍时再临时打开：

```json
"debug_outer_rx": true
```

## 10. 稳定版留档

板端确认稳定后，按 [stable_release_process.md](stable_release_process.md) 保存稳定部署包。

当前推荐格式：

```text
releases/stable_YYYYMMDD_short_name/
  deploy_pkg/
  STABLE_RELEASE.md
```

如果后续新版本异常，直接用稳定版覆盖 SD 卡：

```text
releases/stable_YYYYMMDD_short_name/deploy_pkg/* -> /sdcard/
```
