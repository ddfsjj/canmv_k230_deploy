"""Standalone Gradio page for K230 deploy package generation.

Run this next to a deployed VQ_Estimator instance. It only reads VQ outputs and
writes deploy packages inside this repository.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import zipfile
import argparse
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from vq_deploy import (
    DEFAULT_OUTPUT_RUNTIME,
    ROOT,
    artifact_label,
    build_runtime_from_vq_artifact,
    load_json,
    load_vq_artifact,
    save_json,
    scan_vq_artifacts,
)


APP_CSS = """
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  font-family: "Inter", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
}
.page-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 8px 0 14px;
}
.page-title h2 {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
}
.muted {
  color: #667085;
  font-size: 13px;
}
.artifact-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 14px 16px;
  background: #fff;
}
.artifact-card h3 {
  margin: 0 0 10px;
  font-size: 16px;
}
.artifact-grid {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 6px 12px;
  font-size: 13px;
}
.artifact-grid .k {
  color: #667085;
}
.artifact-grid .v {
  color: #101828;
  overflow-wrap: anywhere;
}
.compact-note {
  padding: 10px 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  color: #475467;
  font-size: 13px;
}
textarea, input, select {
  border-radius: 8px !important;
}
button {
  border-radius: 8px !important;
}
"""


def _local_ip_addresses() -> list[str]:
    addresses: list[str] = []
    hostname = socket.gethostname()
    try:
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127.") and ip not in addresses:
                addresses.append(ip)
    except OSError:
        pass
    return addresses


def _is_port_available(host: str, port: int) -> bool:
    bind_host = host if host not in {"", "localhost"} else "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
        return True
    except OSError:
        return False


def _find_available_port(host: str, start_port: int, try_ports: int) -> int:
    for port in range(start_port, start_port + max(1, try_ports)):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"Cannot find empty port in range: {start_port}-{start_port + max(1, try_ports) - 1}"
    )


def _print_startup_info(host: str, port: int) -> None:
    vq_root = os.environ.get("VQ_K230_OUTPUT_DIR") or default_vq_root()
    print("")
    print("K230 + VQ deploy GUI")
    print(f"  local:   http://127.0.0.1:{port}")
    for ip in _local_ip_addresses():
        print(f"  network: http://{ip}:{port}")
    if host not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        print(f"  host:    http://{host}:{port}")
    print(f"  VQ outputs: {vq_root}")
    print(f"  packages:   {ROOT / 'deploy_pkg'}")
    print("  stop: Ctrl+C")
    print("")


def candidate_vq_roots() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get("VQ_K230_OUTPUT_DIR")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path("/opt/vq/VQ_Estimator/outputs/k230_quant"),
            Path("/opt/vq/VQ_Estimator-gradio-cleanup/outputs/k230_quant"),
            ROOT.parent / "VQ_Estimator" / "outputs" / "k230_quant",
            ROOT.parent / "VQ_Estimator-gradio-cleanup" / "outputs" / "k230_quant",
            Path("/app/outputs/k230_quant"),
            ROOT / "quant_outputs",
        ]
    )
    for path in Path("/opt/vq").glob("*/outputs/k230_quant"):
        candidates.append(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def default_vq_root() -> str:
    for path in candidate_vq_roots():
        if scan_vq_artifacts(path):
            return str(path)
    for path in candidate_vq_roots():
        if path.exists():
            return str(path)
    return str(candidate_vq_roots()[0])


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_int(value: Any, fallback: int | None = None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return fallback
    return int(text)


def parse_float(value: Any, fallback: float | None = None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return fallback
    return float(text)


def parse_bytes(value: str) -> list[int]:
    values = []
    for item in str(value or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item, 0))
    if not values:
        raise ValueError("帧头/帧尾不能为空")
    return values


def format_artifact_info(artifact: dict[str, Any] | None, message: str = "") -> str:
    if artifact is None:
        return f"<div class='compact-note'>{message or '未找到 VQ K230 产物。'}</div>"
    manifest = artifact["manifest"]
    export_cfg = artifact["export_config"]
    data_cfg = export_cfg.get("data", {})
    prep_cfg = export_cfg.get("preprocessing", {})
    summary = artifact.get("compare_summary") or {}
    rows = [
        ("KModel", Path(str(artifact["kmodel"])).name),
        ("Scaler", Path(str(artifact["scaler_json"])).name),
        ("模型类型", manifest.get("model_type", export_cfg.get("model", {}).get("type", ""))),
        ("量化方案", manifest.get("quant_slug", manifest.get("scheme_id", ""))),
        (
            "窗口",
            "base={}, step={}, seq={}, feature={}".format(
                data_cfg.get("base_window_size", 500),
                data_cfg.get("base_step", 200),
                data_cfg.get("sequence_length", 1),
                prep_cfg.get("feature_mode", "raw"),
            ),
        ),
        ("产物目录", str(Path(str(artifact["manifest_path"])).parent)),
    ]
    for key, label in (
        ("total_samples", "样本数"),
        ("kmodel_mae_vs_true", "KMODEL MAE"),
        ("pth_vs_kmodel_mae", "PTH/KMODEL 漂移"),
        ("pth_vs_kmodel_max_abs", "最大漂移"),
    ):
        if key in summary:
            rows.append((label, summary[key]))
    body = "\n".join(
        f"<div class='k'>{key}</div><div class='v'>{value}</div>" for key, value in rows
    )
    return (
        "<div class='artifact-card'>"
        "<h3>当前产物</h3>"
        f"<div class='artifact-grid'>{body}</div>"
        "</div>"
    )


def _search_artifacts(vq_root: str) -> tuple[Path, list[dict[str, Any]], list[Path]]:
    requested = Path(str(vq_root or "").strip()) if str(vq_root or "").strip() else None
    searched: list[Path] = []
    if requested is not None:
        searched.append(requested)
        artifacts = scan_vq_artifacts(requested)
        if artifacts:
            return requested, artifacts, searched
    for path in candidate_vq_roots():
        if requested is not None and str(path) == str(requested):
            continue
        searched.append(path)
        artifacts = scan_vq_artifacts(path)
        if artifacts:
            return path, artifacts, searched
    fallback = requested or candidate_vq_roots()[0]
    return fallback, [], searched


def scan_artifacts(vq_root: str):
    actual_root, artifacts, searched = _search_artifacts(vq_root)
    state = {artifact_label(item): str(item["manifest_path"]) for item in artifacts}
    labels = list(state)
    first = labels[0] if labels else None
    if artifacts:
        info = format_artifact_info(artifacts[0])
    else:
        searched_text = "<br>".join(str(path) for path in searched)
        info = format_artifact_info(
            None,
            "未找到 VQ K230 产物。请先在 VQ 页面完成 K230 量化导出。<br><br>已搜索：<br>" + searched_text,
        )
    return gr.update(value=str(actual_root)), gr.update(choices=labels, value=first), state, info


MODEL_TABLE_HEADERS = ["模型产物", "运行名", "输入通道", "输出槽位", "输出倍率"]
MAPPING_TABLE_HEADERS = ["输入通道", "输出槽位"]


def empty_model_rows() -> list[list[Any]]:
    return []


def default_mapping_rows() -> list[list[Any]]:
    return [["0", "0"]]


def select_artifact(label: str, state: dict[str, str]):
    if not label or label not in state:
        return ("", "", default_mapping_rows())
    artifact = load_vq_artifact(state[label])
    stem = Path(str(artifact["kmodel"])).stem
    return (
        format_artifact_info(artifact),
        stem,
        default_mapping_rows(),
    )


def normalize_mapping_rows(rows: Any) -> list[list[str]]:
    if rows is None:
        return []
    if hasattr(rows, "values"):
        raw_rows = rows.values.tolist()
    elif isinstance(rows, dict) and "data" in rows:
        raw_rows = rows.get("data") or []
    else:
        raw_rows = rows or []

    normalized: list[list[str]] = []
    for row in raw_rows:
        values = list(row)
        if len(values) < 2:
            values.extend([""] * (2 - len(values)))
        input_ch = str(values[0]).strip()
        output_slot = str(values[1]).strip()
        if input_ch or output_slot:
            if not input_ch or not output_slot:
                raise gr.Error("映射表每一行都要同时填写输入通道和输出槽位。")
            normalized.append([input_ch, output_slot])
    return normalized


def add_mapping_row(rows: Any) -> list[list[str]]:
    current = normalize_mapping_rows(rows)
    next_ch = str(len(current))
    current.append([next_ch, next_ch])
    return current


def mapping_one_to_one(channel_count: int) -> list[list[str]]:
    count = int(channel_count or 12)
    return [[str(idx), str(idx)] for idx in range(count)]


def mapping_example_three() -> list[list[str]]:
    return [["0", "2"], ["1", "4"], ["2", "3"]]


def mapping_to_text(rows: Any) -> tuple[str, str]:
    normalized = normalize_mapping_rows(rows)
    if not normalized:
        raise gr.Error("请至少填写一行输入/输出映射。")
    inputs = ",".join(row[0] for row in normalized)
    slots = ",".join("{}:{}".format(row[0], row[1]) for row in normalized)
    return inputs, slots


def add_model_row(
    rows: list[list[Any]] | None,
    artifact_value: str,
    model_name: str,
    mapping_rows: Any,
    output_scale: str,
) -> list[list[Any]]:
    if not artifact_value:
        raise gr.Error("请先选择一个 VQ 产物。")
    current = normalize_model_rows(rows)
    name = str(model_name or "").strip() or "model_{}".format(len(current) + 1)
    input_channels, output_slots = mapping_to_text(mapping_rows)
    current.append([artifact_value, name, input_channels or "0", output_slots or "0", output_scale or "100"])
    return current


def normalize_model_rows(rows: Any) -> list[list[Any]]:
    if rows is None:
        return []
    if hasattr(rows, "values"):
        raw_rows = rows.values.tolist()
    elif isinstance(rows, dict) and "data" in rows:
        raw_rows = rows.get("data") or []
    else:
        raw_rows = rows or []

    normalized: list[list[Any]] = []
    for row in raw_rows:
        values = list(row)
        if len(values) >= 6:
            # Backward compatibility for the earlier table that had an enabled column.
            values = values[1:6]
        if len(values) < 5:
            values.extend([""] * (5 - len(values)))
        values = values[:5]
        if any(str(cell).strip() for cell in values):
            normalized.append(values)
    return normalized


def run_step(title: str, command: list[str]) -> tuple[str, str]:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    output = proc.stdout or ""
    rendered = f"\n== {title} ==\n{output}".rstrip()
    if proc.returncode != 0:
        raise RuntimeError(rendered)
    return output, rendered


def parse_package_root(output: str) -> Path:
    match = re.search(r"deploy package generated:\s*(.+)", output)
    if not match:
        raise RuntimeError("未能从打包输出中识别部署目录。")
    package_root = Path(match.group(1).strip())
    if not package_root.is_absolute():
        package_root = ROOT / package_root
    return package_root


def zip_package(package_root: Path) -> Path:
    zip_path = package_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(package_root))
    return zip_path


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "否"}


def _first_artifact_window(artifact: dict[str, Any]) -> dict[str, Any]:
    export_cfg = artifact["export_config"]
    data_cfg = export_cfg.get("data", {}) if isinstance(export_cfg.get("data", {}), dict) else {}
    prep_cfg = export_cfg.get("preprocessing", {}) if isinstance(export_cfg.get("preprocessing", {}), dict) else {}
    return {
        "base_window_size": int(data_cfg.get("base_window_size", 500)),
        "base_step": int(data_cfg.get("base_step", 200)),
        "sequence_length": int(data_cfg.get("sequence_length", 1)),
        "sequence_step": int(data_cfg.get("sequence_step", 1)),
        "feature_mode": prep_cfg.get("feature_mode", "raw"),
    }


def generate_and_build_multi(
    model_rows: list[list[Any]],
    artifact_state: dict[str, str],
    base_runtime: str,
    output_runtime: str,
    profile_name: str,
    channel_count: int,
    slot_count: int,
    uart_id: int,
    tx_pin: int,
    rx_pin: int,
    baudrate: int,
    header: str,
    tail: str,
    outer_frame_enabled: bool,
    outer_frame_count: int,
    outer_header: str,
    outer_tail: str,
    value_guard_enabled: bool,
    value_min: float,
    value_max: float,
    zero_guard_enabled: bool,
    zero_enter: float,
    zero_exit: float,
    post_enabled: bool,
    post_type: str,
    make_zip: bool,
):
    logs: list[str] = []
    active_rows = [row for row in normalize_model_rows(model_rows) if str(row[0]).strip()]
    if not active_rows:
        raise gr.Error("请先在模型绑定表里至少添加一个模型。")

    base_path = Path(base_runtime)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    output_path = Path(output_runtime)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    updated = load_json(base_path)
    updated["models"] = []
    common_window: dict[str, Any] | None = None
    for index, row in enumerate(active_rows):
        artifact_label_value = str(row[0]).strip()
        if artifact_label_value not in artifact_state:
            raise gr.Error("模型绑定表里的产物不存在，请刷新后重新选择：{}".format(artifact_label_value))
        artifact = load_vq_artifact(artifact_state[artifact_label_value])
        if common_window is None:
            common_window = _first_artifact_window(artifact)
        updated = build_runtime_from_vq_artifact(
            updated,
            artifact,
            append=True,
            model_name=str(row[1] or "").strip() or f"model_{index + 1}",
            output_name=str(row[1] or "").strip() or f"model_{index + 1}",
            profile_name=profile_name or None,
            input_channels_text=str(row[2] or "0"),
            output_slots_text=str(row[3] or "0"),
            output_scale=parse_float(row[4], 100),
            channel_count=int(channel_count),
            slot_count=int(slot_count),
            window_overrides=common_window,
            value_guard={
                "enabled": bool(value_guard_enabled),
                "min": float(value_min),
                "max": float(value_max),
                "replace_non_finite_with": 0.0,
            },
            zero_guard={
                "enabled": bool(zero_guard_enabled),
                "output_value": 0.0,
                "freq_enter_threshold": float(zero_enter),
                "freq_exit_threshold": float(zero_exit),
                "enter_consecutive_windows": 3,
                "exit_consecutive_windows": 3,
                "confidence_absz_threshold": 3.0,
            },
            postprocessing={
                "enabled": bool(post_enabled),
                "type": post_type or "none",
                "exp_smooth_alpha": 0.3,
                "kalman_q": 0.001,
                "kalman_r": 0.1,
                "apply_to_zero_guard": False,
                "reset_on_zero_guard": True,
            },
            uart={
                "enabled": True,
                "uart_id": int(uart_id),
                "tx_pin": int(tx_pin),
                "rx_pin": int(rx_pin),
                "baudrate": int(baudrate),
                "bits": 8,
                "parity": "none",
                "stop": 1,
                "value_type": "int32",
                "byte_order": "big",
                "header": parse_bytes(header),
                "tail": parse_bytes(tail),
                "outer_frame_enabled": bool(outer_frame_enabled),
                "outer_frame_count": int(outer_frame_count),
                "outer_header": parse_bytes(outer_header),
                "outer_tail": parse_bytes(outer_tail),
            },
        )
    save_json(output_path, updated)
    logs.append(f"runtime config generated: {output_path}")

    rel_config = rel_or_abs(output_path)
    _, rendered = run_step("校验运行配置", [sys.executable, "scripts/validate_runtime_config.py", "--config", rel_config])
    logs.append(rendered)
    output, rendered = run_step("生成部署包", [sys.executable, "scripts/make_deploy_package.py", "--config", rel_config])
    logs.append(rendered)
    package_root = parse_package_root(output)
    manifest = package_root / "raw_cnn_k230" / "DEPLOY_MANIFEST.json"
    _, rendered = run_step(
        "校验部署包",
        [sys.executable, "scripts/verify_deploy_package.py", "--manifest", rel_or_abs(manifest)],
    )
    logs.append(rendered)
    zip_path = zip_package(package_root) if make_zip else None
    if zip_path:
        logs.append(f"zip: {zip_path}")
    return "\n".join(logs), str(package_root), str(zip_path or "")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="K230 + VQ 部署包生成器", css=APP_CSS, theme=gr.themes.Soft()) as app:
        gr.HTML(
            "<div class='page-title'>"
            "<h2>K230 + VQ 部署包生成器</h2>"
            "<div class='muted'>读取 VQ 产物，生成 K230 SD 卡部署包</div>"
            "</div>"
        )
        artifact_state = gr.State({})
        gr.HTML(
            "<div class='compact-note'>"
            "流程：扫描 VQ 产物 -> 选择模型产物 -> 填输入/输出 -> 添加绑定 -> 生成部署包。"
            "模型选择只在上方下拉框完成，下面表格只是本次部署要跑的模型清单。"
            "</div>"
        )

        with gr.Row():
            refresh = gr.Button("扫描 VQ 产物", variant="primary", scale=1)
            artifact = gr.Dropdown(label="选择模型产物", choices=[], scale=4)
        artifact_info = gr.HTML()
        with gr.Accordion("产物目录（自动识别，找不到时再改）", open=False):
            vq_root = gr.Textbox(
                label="VQ outputs/k230_quant 目录",
                value=default_vq_root(),
                lines=1,
            )

        gr.Markdown("### 本次部署的模型绑定")
        with gr.Row():
            profile_name = gr.Textbox(label="部署包名称/Profile", value="vq_k230_runtime", lines=1)
            channel_count = gr.Number(label="输入总通道数", value=12, precision=0)
            slot_count = gr.Number(label="输出槽位数", value=12, precision=0)
        with gr.Row():
            model_name = gr.Textbox(label="运行名（可改，用于输出标识）", lines=1)
            output_scale = gr.Textbox(label="输出倍率", value="100", lines=1)
            add_model = gr.Button("添加当前模型绑定", variant="secondary")
        mapping_rows = gr.Dataframe(
            headers=MAPPING_TABLE_HEADERS,
            value=default_mapping_rows(),
            datatype=["str", "str"],
            row_count=(1, "dynamic"),
            col_count=(2, "fixed"),
            label="输入/输出映射表",
            interactive=True,
        )
        with gr.Row():
            add_map = gr.Button("添加一行")
            map_one = gr.Button("12 路一一映射")
            map_three = gr.Button("示例：0→2, 1→4, 2→3")
            map_clear = gr.Button("清空映射")
        model_rows = gr.Dataframe(
            headers=MODEL_TABLE_HEADERS,
            value=empty_model_rows(),
            datatype=["str", "str", "str", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(5, "fixed"),
            label="绑定清单（每行：哪个模型产物、吃哪些输入、输出到哪些槽位）",
            interactive=True,
        )
        clear_models = gr.Button("清空绑定清单")

        with gr.Accordion("UART、保护与高级配置", open=False):
            with gr.Row():
                uart_id = gr.Number(label="UART 号", value=2, precision=0)
                tx_pin = gr.Number(label="TX 引脚", value=11, precision=0)
                rx_pin = gr.Number(label="RX 引脚", value=12, precision=0)
                baudrate = gr.Number(label="波特率", value=921600, precision=0)
            with gr.Accordion("帧协议", open=False):
                with gr.Row():
                    header = gr.Textbox(label="内层帧头", value="85,170", lines=1)
                    tail = gr.Textbox(label="内层帧尾", value="252,207", lines=1)
                    outer_frame_enabled = gr.Checkbox(label="启用外层帧", value=True)
                    outer_frame_count = gr.Number(label="外层帧数量", value=20, precision=0)
                with gr.Row():
                    outer_header = gr.Textbox(label="外层帧头", value="247,127", lines=1)
                    outer_tail = gr.Textbox(label="外层帧尾", value="250,175", lines=1)
            with gr.Row():
                value_guard_enabled = gr.Checkbox(label="输出值保护", value=True)
                value_min = gr.Number(label="输出最小值", value=0.0)
                value_max = gr.Number(label="输出最大值", value=1.0)
            with gr.Row():
                zero_guard_enabled = gr.Checkbox(label="0 干度保护", value=True)
                zero_enter = gr.Number(label="进入阈值", value=480000)
                zero_exit = gr.Number(label="退出阈值", value=500000)
            with gr.Row():
                post_enabled = gr.Checkbox(label="后处理", value=True)
                post_type = gr.Dropdown(label="后处理类型", choices=["kalman", "exp_smooth", "none"], value="kalman")
            with gr.Row():
                base_runtime = gr.Textbox(label="基础 runtime", value="raw_cnn_k230/configs/runtime.json", lines=1)
                output_runtime = gr.Textbox(label="生成的 runtime", value=rel_or_abs(DEFAULT_OUTPUT_RUNTIME), lines=1)

        gr.Markdown("### 生成部署包")
        with gr.Row():
            make_zip = gr.Checkbox(label="生成 zip", value=True, scale=1)
            build = gr.Button("生成配置并打包", variant="primary", scale=3)
        with gr.Row():
            package_root = gr.Textbox(label="部署包目录", lines=1)
            zip_path = gr.Textbox(label="zip 文件", lines=1)
        logs = gr.Textbox(label="执行记录", lines=10)

        refresh.click(scan_artifacts, inputs=[vq_root], outputs=[vq_root, artifact, artifact_state, artifact_info])
        artifact.change(
            select_artifact,
            inputs=[artifact, artifact_state],
            outputs=[
                artifact_info,
                model_name,
                mapping_rows,
            ],
        )
        add_map.click(add_mapping_row, inputs=[mapping_rows], outputs=[mapping_rows])
        map_one.click(mapping_one_to_one, inputs=[channel_count], outputs=[mapping_rows])
        map_three.click(mapping_example_three, outputs=[mapping_rows])
        map_clear.click(lambda: [], outputs=[mapping_rows])
        add_model.click(
            add_model_row,
            inputs=[model_rows, artifact, model_name, mapping_rows, output_scale],
            outputs=[model_rows],
        )
        clear_models.click(lambda: [], outputs=[model_rows])
        build.click(
            generate_and_build_multi,
            inputs=[
                model_rows,
                artifact_state,
                base_runtime,
                output_runtime,
                profile_name,
                channel_count,
                slot_count,
                uart_id,
                tx_pin,
                rx_pin,
                baudrate,
                header,
                tail,
                outer_frame_enabled,
                outer_frame_count,
                outer_header,
                outer_tail,
                value_guard_enabled,
                value_min,
                value_max,
                zero_guard_enabled,
                zero_enter,
                zero_exit,
                post_enabled,
                post_type,
                make_zip,
            ],
            outputs=[logs, package_root, zip_path],
        )
        app.load(scan_artifacts, inputs=[vq_root], outputs=[vq_root, artifact, artifact_state, artifact_info])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the K230 + VQ deploy package GUI.")
    parser.add_argument("--host", default=os.environ.get("K230_DEPLOY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("K230_DEPLOY_PORT", "7861")))
    parser.add_argument(
        "--port-range",
        type=int,
        default=int(os.environ.get("K230_DEPLOY_PORT_RANGE", "20")),
        help="How many ports to try, starting from --port.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if --port is busy instead of trying the next port.",
    )
    parser.add_argument(
        "--vq-root",
        default=os.environ.get("VQ_K230_OUTPUT_DIR", ""),
        help="VQ outputs/k230_quant directory. If omitted, the GUI auto-detects common locations.",
    )
    args = parser.parse_args()

    if args.vq_root:
        os.environ["VQ_K230_OUTPUT_DIR"] = args.vq_root
    port = args.port if args.strict_port else _find_available_port(args.host, args.port, args.port_range)
    os.environ["K230_DEPLOY_PORT"] = str(port)

    _print_startup_info(args.host, port)
    build_app().launch(server_name=args.host, server_port=port)


if __name__ == "__main__":
    main()
