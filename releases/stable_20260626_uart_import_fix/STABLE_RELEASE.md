# Stable Release: 2026-06-26 UART Import Fix

## 状态

这是 `stable_20260626_uart_ok` 的修正版稳定包。

修复内容：

```text
将板端入口中的 `import runtime.app as runtime_app`
改为更兼容 CanMV MicroPython 的：
`from runtime import app as runtime_app`
```

修复原因：

```text
旧写法在 CanMV MicroPython 上可能把 runtime_app 绑定成 runtime 包本身，
导致启动时报：
'module' object has no attribute 'main'
```

## 部署方式

将本目录下的部署包整体拷贝到 SD 卡根目录：

```text
releases/stable_20260626_uart_import_fix/deploy_pkg/* -> /sdcard/
```

## 必须包含

SD 卡根目录：

```text
/sdcard/
  boot.py
  main.py
  raw_cnn_k230/
```

应用目录：

```text
/sdcard/raw_cnn_k230/
  configs/runtime.json
  runtime/
  model/
  run_k230_infer.py
  run_k230_multi_infer.py
  DEPLOY_MANIFEST.json
```

## 已验证

PC 端验证：

```text
python scripts/verify_deploy_package.py
```

入口文件确认：

```text
deploy_pkg/raw_cnn_k230/run_k230_infer.py
deploy_pkg/raw_cnn_k230/run_k230_multi_infer.py
```

必须包含：

```python
from runtime import app as runtime_app
```

不能再出现：

```python
import runtime.app as runtime_app
```

## 配置

```text
profile: cnn_tcn_uart_two_inputs
config: raw_cnn_k230/configs/runtime.json
output.value_guard: enabled, 0.0 ~ 1.0
uart_return_frame: 55 AA + 12 * int32(big endian) + FC CF
frame_len: 52 bytes
```

## 回退方式

如果后续改坏了，使用本修正版稳定包覆盖 SD 卡：

```text
releases/stable_20260626_uart_import_fix/deploy_pkg/* -> /sdcard/
```

