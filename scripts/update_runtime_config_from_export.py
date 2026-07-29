"""从 PC 导出配置更新 K230 runtime.json 的模型条目。"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PC_ROOT = ROOT / "raw_cnn_pc"
K230_ROOT = ROOT / "raw_cnn_k230"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_runtime_config_json(payload: dict) -> str:
    """Format runtime JSON for hand editing.

    Keep channel lists and small protocol arrays on one line, while leaving
    output.slots expanded so slot mappings remain easy to scan.
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    compact_array_keys = (
        "input_channels",
        "header",
        "tail",
        "outer_header",
        "outer_tail",
    )
    for key in compact_array_keys:
        pattern = re.compile(
            r'^(\s*"' + re.escape(key) + r'": )\[\n((?:\s*-?\d+(?:\.\d+)?,?\n)+)(\s*)\]',
            re.MULTILINE,
        )

        def replace(match):
            values = re.findall(r"-?\d+(?:\.\d+)?", match.group(2))
            return "{}[{}]".format(match.group(1), ", ".join(values))

        text = pattern.sub(replace, text)
    return text + "\n"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_runtime_config_json(payload), encoding="utf-8")


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_pc_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PC_ROOT / path


def k230_rel_path_from_pc_export(raw_path: str | Path) -> str:
    resolved = resolve_pc_path(raw_path).resolve()
    try:
        return resolved.relative_to(K230_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "export path must point inside raw_cnn_k230: {}".format(resolved)
        ) from exc


def normalize_model_type(model_type: str) -> str:
    text = str(model_type or "CNN-TCN").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"cnn", "cnn_all", "raw_cnn"}:
        return "cnn"
    if text in {"cnn_lstm", "cnnlstm"}:
        return "cnn_lstm"
    if text in {"cnn_tcn", "cnntcn"}:
        return "cnn_tcn"
    raise ValueError("Unsupported K230 runtime model.type from export config: {}".format(model_type))


def parse_int_list(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None:
        return list(fallback)
    values = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("input channel list must not be empty.")
    return values


def parse_slot_map(raw: str | None, input_channels: list[int], fallback):
    if raw is None:
        return deepcopy(fallback)
    text = str(raw).strip()
    if not text:
        raise ValueError("slot map must not be empty.")
    slots = {}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            channel, slot = item.split(":", 1)
            slots[str(int(channel.strip()))] = int(slot.strip())
        else:
            if len(input_channels) != 1:
                raise ValueError("single slot form is only allowed for one input channel.")
            slots[str(int(input_channels[0]))] = int(item)
    if not slots:
        raise ValueError("slot map must not be empty.")
    return slots


def default_model_name(export_cfg: dict, model_type: str, index: int) -> str:
    raw_name = str(export_cfg.get("name", "")).strip()
    if raw_name:
        return raw_name
    return "model_{}_{}".format(int(index) + 1, model_type)


def build_model_entry(export_cfg: dict, existing_model: dict | None, args, model_index: int) -> dict:
    paths_cfg = export_cfg.get("paths", {})
    data_cfg = export_cfg.get("data", {})
    preprocessing_cfg = export_cfg.get("preprocessing", {})
    model_cfg = export_cfg.get("model", {})

    if "kmodel" not in paths_cfg or "scaler_json" not in paths_cfg:
        raise ValueError("export config paths.kmodel and paths.scaler_json are required.")

    model_type = normalize_model_type(model_cfg.get("type", "CNN-TCN"))
    existing = existing_model or {}
    existing_output = existing.get("output", {}) if isinstance(existing.get("output", {}), dict) else {}

    input_channels = parse_int_list(args.input_channels, existing.get("input_channels", [0]))
    output_slots = parse_slot_map(
        args.output_slots,
        input_channels,
        existing_output.get("slots", {str(input_channels[0]): model_index}),
    )

    model_name = args.model_name or existing.get("name") or default_model_name(export_cfg, model_type, model_index)
    output_name = args.output_name or existing_output.get("name") or model_name

    return {
        "name": str(model_name),
        "enabled": bool(existing.get("enabled", True)),
        "model_type": model_type,
        "input_channels": input_channels,
        "output": {
            "name": str(output_name),
            "slots": output_slots,
            "scale": existing_output.get("scale", 100),
        },
        "assets": {
            "kmodel": k230_rel_path_from_pc_export(paths_cfg["kmodel"]),
            "scaler_json": k230_rel_path_from_pc_export(paths_cfg["scaler_json"]),
        },
        "window": {
            "base_window_size": int(data_cfg.get("base_window_size", 500)),
            "base_step": int(data_cfg.get("base_step", 200)),
            "sequence_length": int(data_cfg.get("sequence_length", 1)),
            "sequence_step": int(data_cfg.get("sequence_step", 1)),
            "feature_mode": preprocessing_cfg.get("feature_mode", "raw"),
        },
    }


def update_runtime_config(runtime_cfg: dict, export_cfg: dict, args) -> dict:
    cfg = deepcopy(runtime_cfg)
    models = list(cfg.get("models", []))
    if args.append:
        model_index = len(models)
        models.append(build_model_entry(export_cfg, None, args, model_index))
    else:
        model_index = int(args.model_index)
        if model_index < 0 or model_index >= len(models):
            raise IndexError("model index {} out of range 0..{}".format(model_index, len(models) - 1))
        models[model_index] = build_model_entry(export_cfg, models[model_index], args, model_index)
    cfg["models"] = models
    if args.profile_name:
        cfg["profile_name"] = args.profile_name
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update K230 runtime.json from a PC export config.")
    parser.add_argument(
        "--export-config",
        default="raw_cnn_pc/configs/export/k230_export_config_cnn_tcn.json",
        help="PC export config path, relative to repo root.",
    )
    parser.add_argument(
        "--runtime-config",
        default="raw_cnn_k230/configs/runtime.json",
        help="K230 runtime config path, relative to repo root.",
    )
    parser.add_argument("--model-index", type=int, default=0, help="Model index to replace when not using --append.")
    parser.add_argument("--append", action="store_true", help="Append as a new model instead of replacing one.")
    parser.add_argument("--model-name", default=None, help="Override runtime models[].name.")
    parser.add_argument("--output-name", default=None, help="Override runtime models[].output.name.")
    parser.add_argument("--input-channels", default=None, help="Comma-separated physical input channels, e.g. 0,1.")
    parser.add_argument("--output-slots", default=None, help="Slot map, e.g. 0:0,1:1 or 0 for one input channel.")
    parser.add_argument("--profile-name", default=None, help="Optional runtime profile_name override.")
    parser.add_argument("--output", default=None, help="Write to another runtime config path instead of in-place.")
    parser.add_argument("--dry-run", action="store_true", help="Print updated JSON without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_path = resolve_repo_path(args.export_config)
    runtime_path = resolve_repo_path(args.runtime_config)
    output_path = resolve_repo_path(args.output) if args.output else runtime_path

    updated = update_runtime_config(load_json(runtime_path), load_json(export_path), args)
    if args.dry_run:
        print(format_runtime_config_json(updated), end="")
        return 0

    save_json(output_path, updated)
    print("runtime config updated:", output_path)
    print("source export config:", export_path)
    print("models:", len(updated.get("models", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
