"""PC 端配置与路径公共工具。"""

import json
from pathlib import Path


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_path(root: Path, raw_path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(root) / path


def require_positive_int(value, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def resolve_positive_step(value, fallback: int, field_name: str) -> int:
    if value is None:
        return require_positive_int(fallback, field_name)
    return require_positive_int(value, field_name)


def resolve_optional_positive_int(value, field_name: str):
    if value is None:
        return None
    return require_positive_int(value, field_name)


def resolve_max_samples(cli_value, config_value):
    # 中文注释：命令行优先；配置为 null 时表示不限制样本数。
    if cli_value is not None:
        return require_positive_int(cli_value, "max_samples")
    return resolve_optional_positive_int(config_value, "runtime.max_samples")


def resolve_calibration_sample_count(value, total: int, field_name: str) -> int:
    # 中文注释：量化校准样本数为 null 时使用当前候选样本全集。
    if total <= 0:
        raise RuntimeError("No scaled samples available for calibration.")
    if value is None:
        return int(total)
    return min(require_positive_int(value, field_name), int(total))
