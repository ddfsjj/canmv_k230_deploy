# Stable Release Process

这份文档用于生成、保存和回退稳定部署包。

## 什么是稳定版

稳定版不是只保存一份 `runtime.json`，而是保存完整部署包：

```text
releases/stable_YYYYMMDD_name/
  deploy_pkg/
  STABLE_RELEASE.md
```

其中：

```text
deploy_pkg/        可以直接拷贝到 SD 卡的完整包
STABLE_RELEASE.md 记录这版的配置、模型、验证结果和回退方式
```

## 生成稳定版

先在开发主线生成部署包：

```bash
python scripts/validate_runtime_config.py
python scripts/run_runtime_sim.py --values 0.23,0.27,0.24,0.21 --json
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

然后建立稳定版目录：

```text
releases/stable_YYYYMMDD_short_name/
```

复制部署包：

```text
deploy_pkg/ -> releases/stable_YYYYMMDD_short_name/deploy_pkg/
```

新增说明文件：

```text
releases/stable_YYYYMMDD_short_name/STABLE_RELEASE.md
```

## STABLE_RELEASE.md 必须记录

至少记录：

```text
日期
状态
部署方式
profile
runtime.json 路径
输出保护范围
返回帧协议
模型资产
PC 验证命令
板端验证结果
回退方式
```

模型资产以 manifest 为准：

```text
releases/stable_YYYYMMDD_short_name/deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

不要靠记忆判断用了哪个 KModel。

## 校验稳定版

稳定版保存后，再校验稳定版自己的 manifest：

```bash
python scripts/verify_deploy_package.py --manifest releases/stable_YYYYMMDD_short_name/deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

必须看到：

```text
deploy package verification ok
```

## 部署稳定版

把稳定版部署包整体拷贝到 SD 卡根目录：

```text
releases/stable_YYYYMMDD_short_name/deploy_pkg/* -> /sdcard/
```

不要只拷贝：

```text
raw_cnn_k230/
```

因为 SD 根目录下的 `main.py` 和 `boot.py` 也属于部署包的一部分。

## 回退稳定版

如果后续新版本改坏了，直接用稳定包覆盖 SD 卡：

```text
releases/stable_YYYYMMDD_short_name/deploy_pkg/* -> /sdcard/
```

当前已保存的稳定版：

```text
releases/stable_20260626_uart_ok/
```

## 什么时候创建新的稳定版

以下情况建议创建新的稳定版：

```text
更换 KModel 或 scaler 后，板端稳定
修改通道或输出槽位后，板端稳定
调整异常阈值后，板端稳定
修改 UART 协议后，板端稳定
运行代码结构变化后，板端稳定
```

不要每次小改都覆盖旧稳定版。创建新目录，保留旧稳定版作为回退点。
