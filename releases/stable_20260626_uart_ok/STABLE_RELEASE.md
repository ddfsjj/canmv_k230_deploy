# Stable Release: 2026-06-26 UART OK

## 废弃说明

这版稳定包已废弃，不建议继续部署。

原因：

```text
run_k230_infer.py / run_k230_multi_infer.py 中使用了：
import runtime.app as runtime_app
```

该写法在 CanMV MicroPython 上可能触发：

```text
'module' object has no attribute 'main'
```

请改用修正版稳定包：

```text
releases/stable_20260626_uart_import_fix/
```

## 状态

板端 UART 在线测试稳定，可以作为当前稳定部署包留档。

## 部署方式

将本目录下的部署包整体拷贝到 SD 卡根目录：

```text
releases/stable_20260626_uart_ok/deploy_pkg/* -> /sdcard/
```

## 运行入口

```text
/sdcard/main.py
  -> /sdcard/raw_cnn_k230/run_k230_infer.py
  -> /sdcard/raw_cnn_k230/runtime/app.py
  -> /sdcard/raw_cnn_k230/configs/runtime.json
```

## 配置

```text
profile: cnn_tcn_uart_two_inputs
config: raw_cnn_k230/configs/runtime.json
output.value_guard: enabled, 0.0 ~ 1.0
uart_return_frame: 55 AA + 12 * int32(big endian) + FC CF
frame_len: 52 bytes
```

## 模型资产

以 `deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json` 为准。本稳定包只包含
`runtime.json` 实际引用的 KModel 和 scaler，不按 `raw_cnn_k230/model/`
目录全量打包。

当前 manifest 记录：

```text
model/cnn-tcn/cnn_tcn_20260625_103918_u8_u8_kld_512.kmodel
model/cnn-tcn/cnn_tcn_20260625_103918_u8_u8_kld_512_scaler.json
```

## 已验证

PC 端验证：

```text
python scripts/validate_runtime_config.py
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --json
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

关键输出验证：

```text
0.23 -> 23 -> 00 00 00 17
0.27 -> 27 -> 00 00 00 1B
0.24 -> 24 -> 00 00 00 18
0.21 -> 21 -> 00 00 00 15
```

输出保护验证：

```text
83886.08 -> 1.0 -> 100 -> 00 00 00 64
```

板端验证：

```text
UART 在线运行稳定
能解析外层大帧
能持续返回 52 字节内层帧
输出保护范围符合当前干度小数协议
```

## 恢复方式

如果后续改坏了，直接用本目录下的稳定部署包覆盖 SD 卡：

```text
releases/stable_20260626_uart_ok/deploy_pkg/* -> /sdcard/
```

覆盖后板端会回到本稳定版本。
