# K230 Raw CNN Deploy

这个仓库现在按“板端方便部署、运行逻辑统一、后续修改集中”的方式组织。

板端主线不再维护单模型和多模型两套逻辑。单模型、多模型都通过
`raw_cnn_k230/configs/runtime.json` 里的 `models[]` 表达。

## 当前结构

```text
输入层：UART/CSV -> 物理通道窗口
推理层：model bindings -> named predictions
输出层：named predictions + channel status -> 12 路返回帧
```

核心目录：

```text
raw_cnn_k230/
  configs/runtime.json        板端唯一运行配置
  runtime/                    统一板端运行框架
    app.py                    启动调度
    online.py                 UART 在线主循环
    csv.py                    CSV 离线/缓存推理
    uart.py                   UART 初始化、帧编码和帧解析
    inputs.py                 UART 读取批次和接收日志
    windows.py                物理通道窗口
    bindings.py               模型绑定和 KPU 推理
    status.py                 异常状态
    protocol.py               异常码常量和返回值打包
    output.py                 12 路返回帧
  run_k230_infer.py           板端主入口
  run_k230_multi_infer.py     兼容入口，内部走同一套 runtime
  model/                      KModel 和 scaler 资产，以 runtime.json/manifest 为准

raw_cnn_pc/
  raw_cnn/                    PC 端公共代码
    models.py                 唯一模型定义层
    data.py                   数据切窗和特征处理
    scaler.py                 scaler 读写和转换
    config.py                 配置公共工具
  infer.py                    PC 推理入口
  build_kmodel.py             ONNX/KModel 导出入口

scripts/
  validate_runtime_config.py  校验 runtime.json
  run_runtime_sim.py          PC 仿真 12 路返回帧
  make_deploy_package.py      生成可直接拷贝到 SD 卡的部署包
  verify_deploy_package.py    校验部署包 manifest 和实际文件

docs/
  architecture.md             运行框架说明
  runtime_config.md           runtime.json 字段和修改入口
  field_change_guide.md       现场换模型、改通道、改槽位操作指南
  deploy.md                   部署流程
  uart_protocol.md            UART 返回帧协议
  anomaly_test.md             异常码串口测试对照
  release_checklist.md        上线检查清单
  stable_release_process.md   稳定版保存和回退流程
  legacy_inventory.md         历史文件和兼容入口清单
```

## 怎么修改

只改运行配置时，优先改：

```text
raw_cnn_k230/configs/runtime.json
```

常见修改位置：

| 目标 | 修改位置 |
| --- | --- |
| 换 KModel | `models[].assets.kmodel` |
| 换 scaler | `models[].assets.scaler_json` |
| 改输入物理通道 | `models[].input_channels` |
| 改输出槽位 | `models[].output.slots` |
| 单模型/多模型切换 | `models[]` 数量和 `enabled` |
| 改窗口长度/步长 | `models[].window` |
| 改异常阈值 | `status.raw_anomaly`、`status.zero_guard`、`status.full_gas_alarm` |
| 改输出保护范围 | `output.value_guard` |
| 改 UART 协议 | `input.uart`、`output.frame` |

模型结构只改：

```text
raw_cnn_pc/raw_cnn/models.py
```

`infer.py` 和 `build_kmodel.py` 不再各自维护模型结构。后续如果新增模型类型，需要先在
`raw_cnn_pc/raw_cnn/models.py` 加模型类和构建函数，再让导出配置或推理配置引用新的
`model.type`。

板端异常码和 12 路返回值只改：

```text
raw_cnn_k230/runtime/status.py
raw_cnn_k230/runtime/output.py
raw_cnn_k230/runtime/protocol.py
```

不要再分别改“单模型脚本”和“多模型脚本”。

## 本地验证

校验当前板端运行配置：

```bash
python scripts/validate_runtime_config.py
```

在 PC 上仿真输出槽位、异常码合成和 52 字节返回帧：

```bash
python scripts/run_runtime_sim.py --json
```

生成部署包：

```bash
python scripts/make_deploy_package.py --clean
```

校验部署包清单和实际文件：

```bash
python scripts/verify_deploy_package.py
```

生成结果：

```text
deploy_pkg/
  boot.py
  main.py
  raw_cnn_k230/
    DEPLOY_MANIFEST.json
    configs/runtime.json
    runtime/
    run_k230_infer.py
    run_k230_multi_infer.py
    model/...
```

## 板端部署

把 `deploy_pkg/` 里的内容整体拷贝到 SD 卡根目录：

```text
deploy_pkg/* -> /sdcard/
```

板端最终结构：

```text
/sdcard/
  boot.py
  main.py
  raw_cnn_k230/
    configs/runtime.json
    runtime/
    model/...
```

`/sdcard/main.py` 是上电自启动入口，只负责进入 `/sdcard/raw_cnn_k230/` 并启动统一
runtime。业务代码仍放在 `raw_cnn_k230/` 里维护。

## 当前 UART 返回帧

当前内层返回帧为 52 字节：

```text
55 AA + 12 * int32 + FC CF
```

当前默认输出槽位：

| Slot | Named prediction |
| --- | --- |
| 0 | `model_1_cnn_tcn_ch0` |
| 1 | `model_1_cnn_tcn_ch1` |
| 2 | `model_2_cnn_tcn_ch0` |
| 3 | `model_2_cnn_tcn_ch1` |
| 4-11 | 空，默认 0 |

异常码跟物理通道走，不跟模型走。比如物理通道 0 异常时，slot 0 和 slot 2 都会带同一个通道状态。

## 上线前检查

上线前至少跑：

```bash
python scripts/validate_runtime_config.py
python scripts/run_runtime_sim.py --json
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

然后检查：

```text
deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

确认里面记录的 profile、KModel、scaler、配置文件和现场要部署的一致。

更多细节见：

- `docs/architecture.md`
- `docs/runtime_config.md`
- `docs/field_change_guide.md`
- `docs/deploy.md`
- `docs/anomaly_test.md`
- `docs/release_checklist.md`
- `docs/stable_release_process.md`
- `docs/legacy_inventory.md`
- `raw_cnn_k230/model/README.md`
- `raw_cnn_pc/README.md`
