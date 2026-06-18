# 脚本功能梳理

## 目录分工

- `raw_cnn_pc/`：PC 端训练后处理、推理验证、ONNX/KModel 导出、量化方案评估和误差定位。
- `raw_cnn_k230/`：K230 板端运行入口，覆盖离线 CSV 验证、串口在线推理、多模型串口输出和串口联调。
- `tools/`：和业务模型相对独立的数据检查、阈值扫描工具。

## PC 端主流程

| 脚本 | 功能 |
|---|---|
| `raw_cnn_pc/infer.py` | PC 端按推理配置加载 `.pth` 和 scaler，对 CSV 数据切窗、标准化并输出预测结果。 |
| `raw_cnn_pc/build_kmodel.py` | PC 端导出部署产物：读取导出配置，生成 ONNX、scaler JSON、校准数据，并调用 nncase 编译 KModel。 |
| `raw_cnn_pc/compare_pth_kmodel.py` | 对同一批样本分别跑 PyTorch `.pth` 和 KModel simulator，输出总体、按 CSV、按干度的误差报告。 |
| `raw_cnn_pc/run_pth_csv_compare_gui.py` | 带 GUI 的 PTH/CSV 对比入口，方便手动选择配置和数据跑验证。 |

## PC 端量化评估

| 脚本 | 功能 |
|---|---|
| `raw_cnn_pc/evaluate_quantization_schemes.py` | 批量评估多套量化方案，为每套方案导出独立 KModel，并生成 Markdown 汇总报告。 |
| `raw_cnn_pc/evaluate_quantization_schemes_custom.py` | 使用外部自定义量化方案列表做评估，适合临时筛选少量方案。 |
| `raw_cnn_pc/evaluate_seg3_quant_schemes.py` | 针对 `cnn_tcn_seg3` 模型的量化方案评估。 |
| `raw_cnn_pc/run_fixed_onnx_quant_matrix.py` | 固定同一份 ONNX，组合不同校准策略和量化精度，比较高干度样本上的误差。 |

## PC 端误差定位

| 脚本 | 功能 |
|---|---|
| `raw_cnn_pc/compare_layerwise_pth_onnx.py` | 导出多输出调试 ONNX，对比 PTH 与 ONNX 中间层输出，判断误差是否在导出阶段出现。 |
| `raw_cnn_pc/compare_pth_onnx_kmodel_gui.py` | GUI 化的 PTH、ONNX、KModel 三路对比入口。 |
| `raw_cnn_pc/compare_cnn_tcn_cutoff_pth_kmodel.py` | 对 CNN-TCN 按指定中间阶段截断导出 KModel，逐层定位 PTH 与 KModel 的误差来源。 |
| `raw_cnn_pc/compare_lstm_cutoff_pth_kmodel.py` | 对 CNN-LSTM 的 LSTM 输入、完整序列、最后时间步等阶段做截断 KModel 对比。 |
| `raw_cnn_pc/compare_lstm_stage_batch.py` | 批量运行 CNN-LSTM 阶段对比，便于一次性比较多个样本或阶段。 |
| `raw_cnn_pc/test_regression_head_sensitivity.py` | 在 PC float CNN-LSTM 上扰动最后时间步特征，评估回归头对特征平移的敏感度。 |

## K230 板端运行

| 脚本 | 功能 |
|---|---|
| `raw_cnn_k230/boot.py` | 板端启动辅助入口，把 `/sdcard/raw_cnn_k230` 加入模块搜索路径。 |
| `raw_cnn_k230/main.py` | 板端上电自启入口，根据 `configs/auto_start_config.json` 选择单模型或多模型入口。 |
| `raw_cnn_k230/run_k230_infer.py` | 板端单模型主入口，支持 CSV 离线验证、串口在线推理、串口 echo、帧回传、调试 ACK 等模式。 |
| `raw_cnn_k230/run_k230_csv_compare.py` | 板端离线 CSV 对比入口，复用 `run_k230_infer.py` 的加载和推理逻辑。 |
| `raw_cnn_k230/run_k230_multi_infer.py` | 板端多模型在线推理入口，按多模型配置输出多路结果。 |
| `raw_cnn_k230/uart_continuous_send_test.py` | 串口连续发送联调工具，用于先验证收发协议和链路稳定性。 |

## 独立工具

| 脚本 | 功能 |
|---|---|
| `tools/check_zero_guard_txt.py` | 检查文本/频率数据在零保护判定规则下的特征和触发情况。 |
| `tools/scan_zero_guard_thresholds.py` | 扫描零保护阈值组合，输出阈值效果统计；当前含本机绝对数据路径，迁移时需要改参数化。 |

## 当前整理结论

- 已删除文件按确认视为真实删除，不恢复。
- 数据目录用途暂不确认，因此配置里的 `../data/new_data` 等路径暂时不改。
- 本轮只修复脚本语法/明显运行阻断点、补 `.gitignore`、增加脚本清单。
