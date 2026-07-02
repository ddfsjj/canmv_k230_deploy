# K230 部署流程

## 校验配置

配置字段说明见 [runtime_config.md](runtime_config.md)。
项目入口边界见 [project_boundary.md](project_boundary.md)。
上线前检查见 [release_checklist.md](release_checklist.md)。

```bash
python scripts/validate_runtime_config.py
```

校验内容包括：

- `runtime.json` 可解析
- 模型文件存在
- scaler 文件存在
- 输出槽位不越界、不冲突
- 输入通道不超过 `input.channel_count`

## 生成部署包

如果刚从 PC 侧导出了新的 `.kmodel` 和 scaler `.json`，可以先用导出配置更新
`runtime.json`：

```bash
python scripts/update_runtime_config_from_export.py --export-config raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json --model-index 0 --dry-run
python scripts/update_runtime_config_from_export.py --export-config raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json --model-index 0
```

`--dry-run` 只打印结果，不写文件。确认模型路径、窗口参数、feature mode 和输出槽位正确后再写回。

```bash
python scripts/make_deploy_package.py --clean
```

## 校验部署包

生成后建议立刻校验 manifest 和实际文件是否一致：

```bash
python scripts/verify_deploy_package.py
```

## PC 仿真输出帧

部署前可以先在 PC 上检查 `runtime.json` 展开后的输出槽位、异常码合成和
52 字节内层返回帧：

```bash
python scripts/run_runtime_sim.py
```

指定 4 路预测值并模拟物理通道 0 异常：

```bash
python scripts/run_runtime_sim.py --values 0.226196,0.265625,0.237549,0.205078 --raw-errors 0=1
```

也可以按 named prediction 指定：

```bash
python scripts/run_runtime_sim.py --pred model_1_cnn_tcn_ch0=0.23
```

输出目录：

```text
deploy_pkg/
```

部署时把 `deploy_pkg/` 里的内容整体拷贝到板端 SD 卡根目录：

```text
deploy_pkg/* -> /sdcard/
```

最终板端结构：

```text
/sdcard/
  boot.py
  main.py
  raw_cnn_k230/
    boot.py
    main.py
    run_k230_infer.py
    run_k230_multi_infer.py
    runtime/
    configs/runtime.json
    model/...
```

`/sdcard/main.py` 是上电自启入口，只负责把 `/sdcard/raw_cnn_k230` 加入
`sys.path`，然后启动统一 runtime。真正业务代码仍放在
`/sdcard/raw_cnn_k230/` 里维护。

## 部署包内容

部署脚本会自动收集：

- SD 卡根目录 `boot.py`
- SD 卡根目录 `main.py`
- 应用目录 `boot.py`
- 应用目录 `main.py`
- `run_k230_infer.py`
- `run_k230_multi_infer.py`
- `runtime/`
- `configs/runtime.json`
- `models[].assets.kmodel`
- `models[].assets.scaler_json`

同时生成：

```text
DEPLOY_MANIFEST.json
```

用于追溯本次部署使用的 profile、模型、scaler 和文件大小。
