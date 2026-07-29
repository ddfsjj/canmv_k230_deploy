"""Helpers for building K230 runtime configs from VQ_Estimator artifacts."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "raw_cnn_k230" / "configs"
DEFAULT_BASE_RUNTIME = CONFIG_DIR / "runtime.json"
DEFAULT_OUTPUT_RUNTIME = CONFIG_DIR / "runtime_vq_generated.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON content must be an object: {path}")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_runtime_config_json(payload), encoding="utf-8")


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def slugify(value: str, fallback: str = "vq_model") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or fallback


def normalize_model_type(model_type: str | None) -> str:
    text = str(model_type or "CNN-TCN").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cnn": "cnn",
        "cnn_all": "cnn",
        "raw_cnn": "cnn",
        "cnn_lstm": "cnn_lstm",
        "cnnlstm": "cnn_lstm",
        "cnn_tcn": "cnn_tcn",
        "cnntcn": "cnn_tcn",
        "cnn_tcn_seg3_soft": "cnn_tcn",
        "cnn_tcn_seg3_soft_stats_moe": "cnn_tcn",
    }
    if text not in aliases:
        raise ValueError(
            "K230 runtime only supports cnn, cnn_tcn and cnn_lstm; got {}".format(model_type)
        )
    return aliases[text]


def parse_int_list(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None:
        return list(fallback)
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("input channel list must not be empty")
    return values


def parse_slot_map(raw: str | None, input_channels: list[int], fallback: dict[str, int]) -> dict[str, int]:
    if raw is None:
        return dict(fallback)
    text = str(raw).strip()
    if not text:
        raise ValueError("output slot map must not be empty")
    slots: dict[str, int] = {}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            channel, slot = item.split(":", 1)
            slots[str(int(channel.strip()))] = int(slot.strip())
        else:
            if len(input_channels) != 1:
                raise ValueError("single slot form is only allowed for one input channel")
            slots[str(int(input_channels[0]))] = int(item)
    if not slots:
        raise ValueError("output slot map must not be empty")
    return slots


def format_runtime_config_json(payload: dict[str, Any]) -> str:
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

        def replace(match: re.Match[str]) -> str:
            values = re.findall(r"-?\d+(?:\.\d+)?", match.group(2))
            return "{}[{}]".format(match.group(1), ", ".join(values))

        text = pattern.sub(replace, text)
    return text + "\n"


def _path_from_text(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _resolve_manifest_sibling(manifest_path: Path, configured: Path | None, fallback_name: str) -> Path:
    if configured is not None and configured.exists():
        return configured
    if configured is not None:
        sibling = manifest_path.parent / configured.name
        if sibling.exists():
            return sibling
    fallback = manifest_path.parent / fallback_name
    if fallback.exists():
        return fallback
    return configured or fallback


def resolve_artifact_file(manifest_path: Path, manifest: dict[str, Any], export_cfg: dict[str, Any], key: str) -> Path:
    paths = export_cfg.get("paths", {}) if isinstance(export_cfg.get("paths", {}), dict) else {}
    configured = _path_from_text(manifest.get(key) or paths.get(key))
    fallback_pattern = "*.kmodel" if key == "kmodel" else "*_scaler.json"
    if configured is not None and configured.exists():
        return configured
    if configured is not None:
        sibling = manifest_path.parent / configured.name
        if sibling.exists():
            return sibling
    matches = sorted(manifest_path.parent.glob(fallback_pattern))
    if matches:
        return matches[0]
    if configured is None:
        raise FileNotFoundError(f"missing {key} in manifest/export config: {manifest_path}")
    return configured


def load_vq_artifact(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"VQ manifest does not exist: {manifest_path}")
    manifest = load_json(manifest_path)
    export_config_path = _resolve_manifest_sibling(
        manifest_path,
        _path_from_text(manifest.get("export_config")),
        "k230_export_config.json",
    )
    export_cfg = load_json(export_config_path)
    kmodel = resolve_artifact_file(manifest_path, manifest, export_cfg, "kmodel")
    scaler_json = resolve_artifact_file(manifest_path, manifest, export_cfg, "scaler_json")
    compare_summary_path = _resolve_manifest_sibling(
        manifest_path,
        _path_from_text(manifest.get("compare_summary")),
        "compare_summary.json",
    )
    compare_summary = load_json(compare_summary_path) if compare_summary_path.exists() else {}
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "export_config_path": export_config_path,
        "export_config": export_cfg,
        "kmodel": kmodel,
        "scaler_json": scaler_json,
        "compare_summary": compare_summary,
    }


def scan_vq_artifacts(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    items: list[dict[str, Any]] = []
    for manifest_path in sorted(root_path.glob("**/manifest.json")):
        try:
            artifact = load_vq_artifact(manifest_path)
            items.append(artifact)
        except Exception:
            continue
    items.sort(key=lambda item: item["manifest_path"].stat().st_mtime, reverse=True)
    return items


def artifact_label(artifact: dict[str, Any]) -> str:
    manifest = artifact["manifest"]
    summary = artifact.get("compare_summary") or {}
    kmodel = artifact["kmodel"]
    model_type = manifest.get("model_type", "")
    scheme = manifest.get("quant_slug") or manifest.get("scheme_id") or ""
    mae = summary.get("kmodel_mae_vs_true")
    drift = summary.get("pth_vs_kmodel_mae")
    metrics = []
    if mae is not None:
        metrics.append(f"MAE={float(mae):.5g}")
    if drift is not None:
        metrics.append(f"drift={float(drift):.5g}")
    suffix = " | " + ", ".join(metrics) if metrics else ""
    return "{} | {} | {}{}".format(kmodel.name, model_type, scheme, suffix)


def summarize_artifact(artifact: dict[str, Any]) -> str:
    manifest = artifact["manifest"]
    export_cfg = artifact["export_config"]
    data_cfg = export_cfg.get("data", {})
    prep_cfg = export_cfg.get("preprocessing", {})
    summary = artifact.get("compare_summary") or {}
    lines = [
        f"manifest: {artifact['manifest_path']}",
        f"kmodel: {artifact['kmodel']}",
        f"scaler_json: {artifact['scaler_json']}",
        f"model_type: {manifest.get('model_type', export_cfg.get('model', {}).get('type', ''))}",
        f"quant: {manifest.get('quant_slug', manifest.get('scheme_id', ''))}",
        "window: base={}, step={}, seq={}, seq_step={}, feature={}".format(
            data_cfg.get("base_window_size", 500),
            data_cfg.get("base_step", 200),
            data_cfg.get("sequence_length", 1),
            data_cfg.get("sequence_step", 1),
            prep_cfg.get("feature_mode", "raw"),
        ),
    ]
    for key in ("total_samples", "kmodel_mae_vs_true", "pth_vs_kmodel_mae", "pth_vs_kmodel_max_abs"):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    return "\n".join(lines)


def _default_slot_map(input_channels: list[int], start_slot: int = 0) -> dict[str, int]:
    return {str(ch): start_slot + idx for idx, ch in enumerate(input_channels)}


def build_runtime_from_vq_artifact(
    base_runtime: dict[str, Any],
    artifact: dict[str, Any],
    *,
    append: bool = False,
    model_index: int = 0,
    model_name: str | None = None,
    output_name: str | None = None,
    profile_name: str | None = None,
    input_channels_text: str | None = None,
    output_slots_text: str | None = None,
    output_scale: float | None = None,
    asset_subdir: str | None = None,
    window_overrides: dict[str, Any] | None = None,
    channel_count: int | None = None,
    slot_count: int | None = None,
    value_guard: dict[str, Any] | None = None,
    zero_guard: dict[str, Any] | None = None,
    postprocessing: dict[str, Any] | None = None,
    uart: dict[str, Any] | None = None,
    runtime_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(base_runtime)
    manifest = artifact["manifest"]
    export_cfg = artifact["export_config"]
    data_cfg = export_cfg.get("data", {}) if isinstance(export_cfg.get("data", {}), dict) else {}
    prep_cfg = export_cfg.get("preprocessing", {}) if isinstance(export_cfg.get("preprocessing", {}), dict) else {}
    model_cfg = export_cfg.get("model", {}) if isinstance(export_cfg.get("model", {}), dict) else {}

    models = list(cfg.get("models", []))
    if append:
        target_index = len(models)
        existing = {}
    else:
        target_index = int(model_index)
        if target_index < 0 or target_index >= len(models):
            raise IndexError(f"model index {target_index} out of range 0..{len(models) - 1}")
        existing = models[target_index]

    existing_output = existing.get("output", {}) if isinstance(existing.get("output", {}), dict) else {}
    input_channels = parse_int_list(input_channels_text, existing.get("input_channels", [0]))
    fallback_slots = existing_output.get("slots")
    if not isinstance(fallback_slots, dict):
        fallback_slots = _default_slot_map(input_channels, target_index)
    output_slots = parse_slot_map(output_slots_text, input_channels, fallback_slots)

    raw_model_name = (
        model_name
        or existing.get("name")
        or manifest.get("quant_slug")
        or Path(str(artifact["kmodel"])).stem
    )
    clean_model_name = slugify(str(raw_model_name), "vq_model")
    clean_output_name = str(output_name or existing_output.get("name") or clean_model_name)

    output_dir_name = asset_subdir or Path(str(artifact["manifest_path"])).parent.name
    asset_dir = "model/vq/{}".format(slugify(output_dir_name, "artifact"))
    kmodel_rel = "{}/{}".format(asset_dir, Path(str(artifact["kmodel"])).name)
    scaler_rel = "{}/{}".format(asset_dir, Path(str(artifact["scaler_json"])).name)

    window = {
        "base_window_size": int(data_cfg.get("base_window_size", 500)),
        "base_step": int(data_cfg.get("base_step", 200)),
        "sequence_length": int(data_cfg.get("sequence_length", 1)),
        "sequence_step": int(data_cfg.get("sequence_step", 1)),
        "feature_mode": prep_cfg.get("feature_mode", "raw"),
    }
    if window_overrides:
        for key, value in window_overrides.items():
            if value not in (None, ""):
                if key == "feature_mode":
                    window[key] = str(value)
                else:
                    window[key] = int(value)

    model_entry = {
        "name": clean_model_name,
        "enabled": bool(existing.get("enabled", True)),
        "model_type": normalize_model_type(str(manifest.get("model_type") or model_cfg.get("type") or "CNN-TCN")),
        "input_channels": input_channels,
        "output": {
            "name": clean_output_name,
            "slots": output_slots,
            "scale": float(output_scale if output_scale is not None else existing_output.get("scale", 100)),
        },
        "assets": {
            "kmodel": kmodel_rel,
            "scaler_json": scaler_rel,
            "kmodel_source": str(Path(str(artifact["kmodel"])).resolve()),
            "scaler_json_source": str(Path(str(artifact["scaler_json"])).resolve()),
            "source_manifest": str(Path(str(artifact["manifest_path"])).resolve()),
            "source_export_config": str(Path(str(artifact["export_config_path"])).resolve()),
        },
        "window": window,
    }

    if append:
        models.append(model_entry)
    else:
        models[target_index] = model_entry
    cfg["models"] = models

    if profile_name:
        cfg["profile_name"] = str(profile_name)
    if channel_count is not None:
        cfg.setdefault("input", {})["channel_count"] = int(channel_count)
    if slot_count is not None:
        cfg.setdefault("output", {})["slot_count"] = int(slot_count)
    if value_guard is not None:
        cfg.setdefault("output", {})["value_guard"] = value_guard
    if zero_guard is not None:
        cfg.setdefault("status", {})["zero_guard"] = zero_guard
    if postprocessing is not None:
        cfg.setdefault("status", {})["postprocessing"] = postprocessing
    if uart:
        cfg.setdefault("input", {}).setdefault("uart", {}).update(uart)
        cfg.setdefault("output", {}).setdefault("frame", {}).update(
            {
                key: value
                for key, value in uart.items()
                if key in {"outer_frame_enabled", "outer_frame_count", "outer_header", "outer_tail"}
            }
        )
    if runtime_flags:
        cfg.setdefault("runtime", {}).update(runtime_flags)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create K230 runtime config from a VQ_Estimator artifact.")
    parser.add_argument("--manifest", required=True, help="VQ artifact manifest.json path.")
    parser.add_argument("--base-runtime", default=str(DEFAULT_BASE_RUNTIME), help="Base runtime config.")
    parser.add_argument("--output-runtime", default=str(DEFAULT_OUTPUT_RUNTIME), help="Output runtime config.")
    parser.add_argument("--append", action="store_true", help="Append model instead of replacing --model-index.")
    parser.add_argument("--model-index", type=int, default=0, help="Model index to replace.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--profile-name", default=None)
    parser.add_argument("--input-channels", default=None, help="Comma-separated channels, e.g. 0,1,2.")
    parser.add_argument("--output-slots", default=None, help="Slot map, e.g. 0:0,1:1 or 0.")
    parser.add_argument("--output-scale", type=float, default=None)
    parser.add_argument("--asset-subdir", default=None)
    parser.add_argument("--channel-count", type=int, default=None)
    parser.add_argument("--slot-count", type=int, default=None)
    parser.add_argument("--base-window-size", type=int, default=None)
    parser.add_argument("--base-step", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-step", type=int, default=None)
    parser.add_argument("--feature-mode", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = load_vq_artifact(args.manifest)
    base_runtime = load_json(resolve_repo_path(args.base_runtime))
    window_overrides = {
        "base_window_size": args.base_window_size,
        "base_step": args.base_step,
        "sequence_length": args.sequence_length,
        "sequence_step": args.sequence_step,
        "feature_mode": args.feature_mode,
    }
    updated = build_runtime_from_vq_artifact(
        base_runtime,
        artifact,
        append=args.append,
        model_index=args.model_index,
        model_name=args.model_name,
        output_name=args.output_name,
        profile_name=args.profile_name,
        input_channels_text=args.input_channels,
        output_slots_text=args.output_slots,
        output_scale=args.output_scale,
        asset_subdir=args.asset_subdir,
        window_overrides=window_overrides,
        channel_count=args.channel_count,
        slot_count=args.slot_count,
    )
    if args.dry_run:
        print(format_runtime_config_json(updated), end="")
        return 0
    output_path = resolve_repo_path(args.output_runtime)
    save_json(output_path, updated)
    print("runtime config generated:", output_path)
    print("source manifest:", artifact["manifest_path"])
    print("kmodel:", artifact["kmodel"])
    print("scaler_json:", artifact["scaler_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
