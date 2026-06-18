"""
大将军，这个脚本是“自定义量化方案批量评估入口”。

它和老的 `evaluate_quantization_schemes.py` 的关系是：
1. 老脚本内置了一批固定方案，更适合历史通用评估。
2. 这个新脚本不再把方案写死在 Python 里，而是从外部 json 读取。
3. 这样当我们只想针对某一版模型，例如 `cnn_lstm_20260320_023445`，
   单独试几套量化组合时，不需要继续改老脚本源码。

这个文件主要解决两个问题：
1. 让“方案定义”和“评估执行”分离。
2. 让不同模型、不同数据子集都能复用同一套评估流程。

典型使用场景：
1. 想对高干度子集单独试 3 套量化方案。
2. 想保留旧版 kmodel，同时再生成 uniform / noclip 等新版 kmodel。
3. 想把评估结果统一输出到独立目录，方便后续比较和写报告。

这个脚本做的事情依次是：
1. 读取 `--schemes_json` 里的方案列表。
2. 基于基础导出配置，给每个方案展开出独立的 export_config。
3. 调用已有构建逻辑生成 onnx / kmodel / 校准产物。
4. 调用已有 compare 逻辑跑 `.pth vs kmodel` 对比。
5. 最后复用老脚本的 markdown 报告生成逻辑。

要点提醒：
1. 这个脚本本身不定义量化参数，只负责“调度”。
2. 真正的方案内容在 `quant_schemes_*.json` 里。
3. 真正的导出与对比实现仍然复用老脚本，避免出现两套不一致逻辑。
"""

import argparse
from copy import deepcopy
from pathlib import Path

import evaluate_quantization_schemes as base


def parse_args():
    # 这里只保留最核心的参数，避免命令行入口变得过于分散。
    parser = argparse.ArgumentParser(description="Evaluate custom K230 quantization schemes from json list.")
    parser.add_argument("--infer_config", type=str, default="infer_config.json")
    parser.add_argument("--export_config", type=str, default="k230_export_config.json")
    parser.add_argument("--schemes_json", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scheme_eval_custom")
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_dryness", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=2000)
    return parser.parse_args()


def load_scheme_list(path: Path):
    # 自定义方案列表走独立入口，避免修改历史评估脚本带来额外风险。
    # 同时这里故意做严格校验，防止 json 里漏字段导致跑到中途才报错。
    payload = base.load_json(path)
    if isinstance(payload, dict):
        payload = payload.get("schemes", None)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid schemes json, expected non-empty list or object with `schemes`: {path}")
    required = {
        "id",
        "title",
        "samples_count",
        "sampling_strategy",
        "quant_type",
        "weight_quant_type",
        "calibrate_method",
    }
    out = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid scheme at index {idx}: expected object")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(
                "Scheme {} missing fields: {}".format(item.get("id", idx), ", ".join(missing))
            )
        out.append(item)
    return out


def main():
    # 主流程只做“读方案 -> 逐方案构建 -> 逐方案对比 -> 生成总报告”。
    # 这样后面大将军如果继续新增方案，只需要改 json，不需要再碰执行逻辑。
    args = parse_args()
    root = Path(__file__).resolve().parent
    infer_cfg_path = (root / args.infer_config).resolve() if not Path(args.infer_config).is_absolute() else Path(args.infer_config)
    export_cfg_path = (root / args.export_config).resolve() if not Path(args.export_config).is_absolute() else Path(args.export_config)
    schemes_path = (root / args.schemes_json).resolve() if not Path(args.schemes_json).is_absolute() else Path(args.schemes_json)
    data_dir = Path(args.data_dir).resolve()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = load_scheme_list(schemes_path)
    if args.scheme:
        schemes = [scheme for scheme in schemes if scheme["id"] == args.scheme]
        if not schemes:
            raise KeyError(f"Unknown scheme id: {args.scheme}")

    base_export_cfg = base.load_json(export_cfg_path)
    for scheme in schemes:
        scheme_export_cfg = base.make_scheme_export_cfg(deepcopy(base_export_cfg), scheme)
        scheme_cfg_path = output_dir / f"{scheme['id']}_export_config.json"
        base.save_json(scheme_cfg_path, scheme_export_cfg)

        print("=== Build Scheme ===")
        print("scheme:", scheme["id"])
        build_info = base.build_scheme_model(root, scheme_export_cfg)
        base.save_json(output_dir / f"{scheme['id']}_build_info.json", build_info)

        print("=== Compare Scheme ===")
        print("scheme:", scheme["id"])
        base.run_scheme_compare(
            root,
            infer_cfg_path,
            scheme_cfg_path,
            data_dir,
            output_dir,
            scheme,
            max_samples=args.max_samples,
            max_per_dryness=args.max_per_dryness,
            log_every=args.log_every,
        )

    report_text = base.build_markdown_report(output_dir)
    report_path = output_dir / "quantization_scheme_report.md"
    base.save_text(report_path, report_text)
    print("report_path:", report_path)


if __name__ == "__main__":
    main()
