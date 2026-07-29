"""校验 K230 统一运行配置。

这个脚本在 PC 端执行，用于部署前检查 runtime.json 是否完整可用。
"""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K230_DIR = ROOT / "raw_cnn_k230"
if str(K230_DIR) not in sys.path:
    sys.path.insert(0, str(K230_DIR))

from runtime.config import is_new_runtime_config, load_runtime_config, to_legacy_multi_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Validate raw_cnn_k230/configs/runtime.json.")
    parser.add_argument(
        "--config",
        default="raw_cnn_k230/configs/runtime.json",
        help="Path to runtime config, relative to repo root by default.",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_asset_path(base_dir, rel_path, source_path=None):
    if source_path:
        source = Path(source_path)
        if not source.is_absolute():
            source = ROOT / source
        if source.exists():
            return source
    return base_dir / rel_path


def require_file(base_dir, rel_path, label, errors, source_path=None):
    path = resolve_asset_path(base_dir, rel_path, source_path)
    if not path.exists():
        errors.append(f"{label} missing: {rel_path} -> {path}")
    elif not path.is_file():
        errors.append(f"{label} is not a file: {rel_path} -> {path}")
    return path


def validate_files(cfg, errors):
    if isinstance(cfg, dict) and isinstance(cfg.get("models"), list):
        for idx, model in enumerate(cfg.get("models", [])):
            assets = model.get("assets", {}) if isinstance(model, dict) else {}
            if assets:
                require_file(
                    K230_DIR,
                    assets.get("kmodel", ""),
                    f"models[{idx}].kmodel",
                    errors,
                    assets.get("kmodel_source"),
                )
                require_file(
                    K230_DIR,
                    assets.get("scaler_json", ""),
                    f"models[{idx}].scaler_json",
                    errors,
                    assets.get("scaler_json_source"),
                )
        return

    legacy = to_legacy_multi_config(cfg)
    for idx, model in enumerate(legacy.get("models", [])):
        paths = model.get("paths", {})
        require_file(K230_DIR, paths.get("kmodel", ""), f"models[{idx}].kmodel", errors)
        require_file(K230_DIR, paths.get("scaler_json", ""), f"models[{idx}].scaler_json", errors)


def validate_slots(cfg, errors):
    legacy = to_legacy_multi_config(cfg)
    output_cfg = legacy.get("runtime", {}).get("output", {})
    slots = output_cfg.get("slots", [])
    value_count = int(legacy.get("uart", {}).get("value_count", len(slots) or 12))
    if len(slots) > value_count:
        errors.append(f"output slots length {len(slots)} exceeds value_count {value_count}")
    names = set()
    for idx, name in enumerate(slots):
        if name is None:
            continue
        text = str(name)
        if text in names:
            errors.append(f"duplicate output name in slots: {text}")
        names.add(text)
        if idx >= value_count:
            errors.append(f"output slot {idx} out of range value_count {value_count}")


def validate_output_guard(cfg, errors):
    output_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}
    guard_cfg = output_cfg.get("value_guard", {})
    if not guard_cfg:
        return
    if not isinstance(guard_cfg, dict):
        errors.append("output.value_guard must be an object")
        return
    min_value = guard_cfg.get("min", None)
    max_value = guard_cfg.get("max", None)
    try:
        if min_value is not None and max_value is not None and float(min_value) > float(max_value):
            errors.append("output.value_guard.min must be <= output.value_guard.max")
    except Exception:
        errors.append("output.value_guard.min/max must be numeric when present")
    try:
        float(guard_cfg.get("replace_non_finite_with", 0.0))
    except Exception:
        errors.append("output.value_guard.replace_non_finite_with must be numeric")


def main():
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    errors = []
    try:
        cfg = load_runtime_config(str(config_path))
    except Exception as exc:
        raise SystemExit(f"Config parse/shape error: {exc}") from exc

    if not is_new_runtime_config(cfg):
        print("WARN: legacy config detected; file checks still run against converted shape.")

    validate_files(cfg, errors)
    validate_slots(cfg, errors)
    validate_output_guard(cfg, errors)

    if errors:
        print("runtime config validation failed:")
        for item in errors:
            print("-", item)
        raise SystemExit(1)

    legacy = to_legacy_multi_config(cfg)
    print("runtime config validation ok")
    print("config:", config_path)
    print("profile:", legacy.get("name", ""))
    print("models:", len(legacy.get("models", [])))
    print("uart_value_count:", legacy.get("uart", {}).get("value_count", 12))
    print("legacy_preview:")
    print(json.dumps({"runtime": legacy.get("runtime", {}), "inputs": legacy.get("inputs", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
