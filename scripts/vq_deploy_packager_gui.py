"""GUI for generating K230 deploy packages from VQ_Estimator artifacts."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from tkinter import BooleanVar, END, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

from vq_deploy import (
    DEFAULT_OUTPUT_RUNTIME,
    ROOT,
    artifact_label,
    build_runtime_from_vq_artifact,
    load_json,
    load_vq_artifact,
    save_json,
    scan_vq_artifacts,
    summarize_artifact,
)


CONFIG_DIR = ROOT / "raw_cnn_k230" / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "runtime.json"


def _default_vq_root() -> str:
    env_value = os.environ.get("VQ_K230_OUTPUT_DIR")
    if env_value:
        return env_value
    sibling = ROOT.parent / "VQ_Estimator" / "outputs" / "k230_quant"
    if sibling.exists():
        return str(sibling)
    local = ROOT / "quant_outputs"
    if local.exists():
        return str(local)
    return str(sibling)


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_optional_int(text: str) -> int | None:
    text = str(text or "").strip()
    return int(text) if text else None


def _parse_optional_float(text: str) -> float | None:
    text = str(text or "").strip()
    return float(text) if text else None


def _parse_bool(value: bool) -> bool:
    return bool(value)


def _parse_byte_list(text: str) -> list[int]:
    values = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item, 0))
    if not values:
        raise ValueError("帧头/帧尾不能为空")
    return values


class VqDeployPackagerGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("K230 + VQ 部署包生成器")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.artifacts_by_label: dict[str, dict] = {}
        self.selected_artifact: dict | None = None
        self.last_package_root: Path | None = None
        self.last_zip_path: Path | None = None

        self.vq_root_var = StringVar(value=_default_vq_root())
        self.artifact_var = StringVar(value="")
        self.base_config_var = StringVar(value=self._default_config_text())
        self.output_config_var = StringVar(value=_rel_or_abs(DEFAULT_OUTPUT_RUNTIME))
        self.profile_var = StringVar(value="vq_k230_runtime")
        self.model_name_var = StringVar(value="")
        self.output_name_var = StringVar(value="")
        self.append_var = BooleanVar(value=False)
        self.model_index_var = StringVar(value="0")
        self.input_channels_var = StringVar(value="0")
        self.output_slots_var = StringVar(value="0")
        self.output_scale_var = StringVar(value="100")
        self.asset_subdir_var = StringVar(value="")
        self.channel_count_var = StringVar(value="12")
        self.slot_count_var = StringVar(value="12")
        self.base_window_var = StringVar(value="")
        self.base_step_var = StringVar(value="")
        self.seq_len_var = StringVar(value="")
        self.seq_step_var = StringVar(value="")
        self.feature_mode_var = StringVar(value="")

        self.uart_id_var = StringVar(value="2")
        self.tx_pin_var = StringVar(value="11")
        self.rx_pin_var = StringVar(value="12")
        self.baudrate_var = StringVar(value="921600")
        self.outer_frame_var = BooleanVar(value=True)
        self.outer_count_var = StringVar(value="20")
        self.header_var = StringVar(value="85,170")
        self.tail_var = StringVar(value="252,207")
        self.outer_header_var = StringVar(value="247,127")
        self.outer_tail_var = StringVar(value="250,175")

        self.value_guard_var = BooleanVar(value=True)
        self.value_min_var = StringVar(value="0.0")
        self.value_max_var = StringVar(value="1.0")
        self.zero_guard_var = BooleanVar(value=True)
        self.zero_enter_var = StringVar(value="480000")
        self.zero_exit_var = StringVar(value="500000")
        self.zero_enter_count_var = StringVar(value="3")
        self.zero_exit_count_var = StringVar(value="3")
        self.post_enabled_var = BooleanVar(value=True)
        self.post_type_var = StringVar(value="kalman")
        self.kalman_q_var = StringVar(value="0.001")
        self.kalman_r_var = StringVar(value="0.1")
        self.smooth_alpha_var = StringVar(value="0.3")
        self.debug_trace_var = BooleanVar(value=True)
        self.debug_tx_var = BooleanVar(value=True)
        self.quiet_var = BooleanVar(value=False)

        self.make_zip_var = BooleanVar(value=True)
        self.status_var = StringVar(value="请选择 VQ 产物，生成 runtime 配置，再生成部署包。")

        self._build_ui()
        self._scan_vq_artifacts()
        self._poll_messages()

    def _default_config_text(self) -> str:
        if DEFAULT_CONFIG.exists():
            return _rel_or_abs(DEFAULT_CONFIG)
        return ""

    def _config_choices(self) -> list[str]:
        if not CONFIG_DIR.exists():
            return []
        return [_rel_or_abs(path) for path in sorted(CONFIG_DIR.glob("runtime*.json"))]

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text="K230 + VQ 部署包生成器", font=("Microsoft YaHei UI", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.notebook = ttk.Notebook(outer)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(12, 8))
        self._build_artifact_tab()
        self._build_model_tab()
        self._build_runtime_tab()
        self._build_package_tab()

        ttk.Label(outer, textvariable=self.status_var).grid(row=2, column=0, sticky="ew")

    def _build_artifact_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="VQ 产物")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)

        ttk.Label(tab, text="VQ 输出目录").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(tab, textvariable=self.vq_root_var).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(tab, text="选择...", command=self.choose_vq_root).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(tab, text="刷新", command=self._scan_vq_artifacts).grid(row=0, column=3)

        ttk.Label(tab, text="可用产物").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(10, 0))
        self.artifact_combo = ttk.Combobox(tab, textvariable=self.artifact_var, state="readonly")
        self.artifact_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        self.artifact_combo.bind("<<ComboboxSelected>>", lambda event: self._select_artifact())

        ttk.Button(tab, text="选择 manifest...", command=self.choose_manifest).grid(
            row=2, column=1, sticky="w", pady=(10, 8)
        )

        info_frame = ttk.LabelFrame(tab, text="产物信息", padding=8)
        info_frame.grid(row=3, column=0, columnspan=4, sticky="nsew")
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        self.artifact_text = self._make_log_text(info_frame, height=22)
        self.artifact_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.artifact_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.artifact_text.configure(yscrollcommand=scrollbar.set)

    def _build_model_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="模型与通道")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        self._entry_row(tab, 0, "基础 runtime", self.base_config_var, button=("选择...", self.choose_base_config))
        self._entry_row(tab, 1, "输出 runtime", self.output_config_var, button=("保存为...", self.choose_output_config))
        self._entry_row(tab, 2, "Profile 名", self.profile_var)
        self._entry_row(tab, 3, "模型名", self.model_name_var)
        self._entry_row(tab, 4, "输出名", self.output_name_var)

        ttk.Checkbutton(tab, text="追加模型", variable=self.append_var).grid(row=5, column=1, sticky="w", pady=5)
        self._entry_cell(tab, 5, 2, "替换模型序号", self.model_index_var)

        self._entry_cell(tab, 6, 0, "输入通道", self.input_channels_var)
        self._entry_cell(tab, 6, 2, "输出槽位", self.output_slots_var)
        self._entry_cell(tab, 7, 0, "输入总通道数", self.channel_count_var)
        self._entry_cell(tab, 7, 2, "输出槽位数", self.slot_count_var)
        self._entry_cell(tab, 8, 0, "输出倍率", self.output_scale_var)
        self._entry_cell(tab, 8, 2, "部署模型子目录", self.asset_subdir_var)

        ttk.Separator(tab).grid(row=9, column=0, columnspan=4, sticky="ew", pady=12)
        self._entry_cell(tab, 10, 0, "base_window_size", self.base_window_var)
        self._entry_cell(tab, 10, 2, "base_step", self.base_step_var)
        self._entry_cell(tab, 11, 0, "sequence_length", self.seq_len_var)
        self._entry_cell(tab, 11, 2, "sequence_step", self.seq_step_var)
        self._entry_row(tab, 12, "feature_mode", self.feature_mode_var)

        ttk.Button(tab, text="生成 runtime 配置", command=self.generate_runtime_config).grid(
            row=13, column=1, sticky="w", pady=(16, 0)
        )

    def _build_runtime_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="UART 与保护")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        self._entry_cell(tab, 0, 0, "UART 号", self.uart_id_var)
        self._entry_cell(tab, 0, 2, "波特率", self.baudrate_var)
        self._entry_cell(tab, 1, 0, "TX 引脚", self.tx_pin_var)
        self._entry_cell(tab, 1, 2, "RX 引脚", self.rx_pin_var)
        self._entry_cell(tab, 2, 0, "内层帧头", self.header_var)
        self._entry_cell(tab, 2, 2, "内层帧尾", self.tail_var)
        ttk.Checkbutton(tab, text="启用外层帧", variable=self.outer_frame_var).grid(row=3, column=1, sticky="w", pady=5)
        self._entry_cell(tab, 3, 2, "外层帧数量", self.outer_count_var)
        self._entry_cell(tab, 4, 0, "外层帧头", self.outer_header_var)
        self._entry_cell(tab, 4, 2, "外层帧尾", self.outer_tail_var)

        ttk.Separator(tab).grid(row=5, column=0, columnspan=4, sticky="ew", pady=12)
        ttk.Checkbutton(tab, text="输出值保护", variable=self.value_guard_var).grid(row=6, column=1, sticky="w")
        self._entry_cell(tab, 6, 2, "最小/最大", self.value_min_var, width=12)
        ttk.Entry(tab, textvariable=self.value_max_var, width=12).grid(row=6, column=4, sticky="w")

        ttk.Checkbutton(tab, text="0 干度保护", variable=self.zero_guard_var).grid(row=7, column=1, sticky="w", pady=5)
        self._entry_cell(tab, 7, 2, "进入阈值", self.zero_enter_var)
        self._entry_cell(tab, 8, 0, "退出阈值", self.zero_exit_var)
        self._entry_cell(tab, 8, 2, "进入/退出次数", self.zero_enter_count_var, width=12)
        ttk.Entry(tab, textvariable=self.zero_exit_count_var, width=12).grid(row=8, column=4, sticky="w")

        ttk.Checkbutton(tab, text="后处理", variable=self.post_enabled_var).grid(row=9, column=1, sticky="w", pady=5)
        self._entry_cell(tab, 9, 2, "类型", self.post_type_var)
        self._entry_cell(tab, 10, 0, "Kalman Q", self.kalman_q_var)
        self._entry_cell(tab, 10, 2, "Kalman R", self.kalman_r_var)
        self._entry_cell(tab, 11, 0, "平滑 alpha", self.smooth_alpha_var)

        ttk.Separator(tab).grid(row=12, column=0, columnspan=4, sticky="ew", pady=12)
        ttk.Checkbutton(tab, text="打印预测追踪", variable=self.debug_trace_var).grid(row=13, column=1, sticky="w")
        ttk.Checkbutton(tab, text="打印发送耗时", variable=self.debug_tx_var).grid(row=13, column=2, sticky="w")
        ttk.Checkbutton(tab, text="安静模式", variable=self.quiet_var).grid(row=13, column=3, sticky="w")

    def _build_package_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="生成部署包")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        options = ttk.Frame(tab)
        options.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(options, text="生成 zip", variable=self.make_zip_var).grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Button(options, text="生成配置并打包", command=self.generate_and_build).grid(row=0, column=1, padx=(0, 8))
        self.start_button = ttk.Button(options, text="仅按当前配置打包", command=self.start_build)
        self.start_button.grid(row=0, column=2, padx=(0, 8))
        self.open_folder_button = ttk.Button(options, text="打开输出目录", command=self.open_output_folder, state="disabled")
        self.open_folder_button.grid(row=0, column=3, padx=(0, 8))
        self.open_zip_button = ttk.Button(options, text="打开 zip 位置", command=self.open_zip_folder, state="disabled")
        self.open_zip_button.grid(row=0, column=4)

        log_frame = ttk.LabelFrame(tab, text="执行记录", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = self._make_log_text(log_frame, height=24)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _entry_row(self, parent, row: int, label: str, var: StringVar, button=None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        if button:
            text, command = button
            ttk.Button(parent, text=text, command=command).grid(row=row, column=3, sticky="w", padx=(8, 0), pady=4)

    def _entry_cell(self, parent, row: int, col: int, label: str, var: StringVar, width: int = 18) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="ew", pady=4)

    def _make_log_text(self, parent, height=18):
        from tkinter import Text

        text = Text(parent, height=height, wrap="word", font=("Consolas", 10))
        text.configure(state="disabled")
        return text

    def choose_vq_root(self) -> None:
        selected = filedialog.askdirectory(title="选择 VQ outputs/k230_quant 目录", initialdir=self.vq_root_var.get())
        if selected:
            self.vq_root_var.set(selected)
            self._scan_vq_artifacts()

    def choose_manifest(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 VQ manifest.json",
            initialdir=self.vq_root_var.get(),
            filetypes=[("manifest.json", "manifest.json"), ("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            artifact = load_vq_artifact(selected)
            label = artifact_label(artifact)
            self.artifacts_by_label[label] = artifact
            self.artifact_combo.configure(values=list(self.artifacts_by_label))
            self.artifact_var.set(label)
            self._select_artifact()

    def choose_base_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择基础 runtime 配置",
            initialdir=str(CONFIG_DIR),
            filetypes=[("JSON 配置", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.base_config_var.set(_rel_or_abs(Path(selected)))

    def choose_output_config(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="保存生成的 runtime 配置",
            initialdir=str(CONFIG_DIR),
            initialfile="runtime_vq_generated.json",
            defaultextension=".json",
            filetypes=[("JSON 配置", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.output_config_var.set(_rel_or_abs(Path(selected)))

    def _scan_vq_artifacts(self) -> None:
        root = Path(self.vq_root_var.get().strip())
        artifacts = scan_vq_artifacts(root)
        self.artifacts_by_label = {artifact_label(item): item for item in artifacts}
        labels = list(self.artifacts_by_label)
        self.artifact_combo.configure(values=labels)
        if labels:
            self.artifact_var.set(labels[0])
            self._select_artifact()
            self.status_var.set("已读取 {} 个 VQ 产物。".format(len(labels)))
        else:
            self.artifact_var.set("")
            self.selected_artifact = None
            self._set_text(self.artifact_text, "未找到 manifest.json。")
            self.status_var.set("未找到 VQ 产物，请检查输出目录。")

    def _select_artifact(self) -> None:
        label = self.artifact_var.get()
        artifact = self.artifacts_by_label.get(label)
        if not artifact:
            return
        self.selected_artifact = artifact
        manifest = artifact["manifest"]
        export_cfg = artifact["export_config"]
        data_cfg = export_cfg.get("data", {})
        prep_cfg = export_cfg.get("preprocessing", {})
        default_name = Path(str(artifact["kmodel"])).stem
        self.model_name_var.set(default_name)
        self.output_name_var.set(default_name)
        self.profile_var.set(default_name)
        self.asset_subdir_var.set(Path(str(artifact["manifest_path"])).parent.name)
        self.base_window_var.set(str(data_cfg.get("base_window_size", "")))
        self.base_step_var.set(str(data_cfg.get("base_step", "")))
        self.seq_len_var.set(str(data_cfg.get("sequence_length", "")))
        self.seq_step_var.set(str(data_cfg.get("sequence_step", "")))
        self.feature_mode_var.set(str(prep_cfg.get("feature_mode", "")))
        self._set_text(self.artifact_text, summarize_artifact(artifact))

    def _runtime_config_path(self) -> Path:
        text = self.output_config_var.get().strip().strip('"')
        path = Path(text)
        if not path.is_absolute():
            path = ROOT / path
        return path

    def _base_config_path(self) -> Path:
        text = self.base_config_var.get().strip().strip('"')
        path = Path(text)
        if not path.is_absolute():
            path = ROOT / path
        return path

    def _build_runtime_options(self) -> dict:
        value_guard = {
            "enabled": _parse_bool(self.value_guard_var.get()),
            "min": float(self.value_min_var.get()),
            "max": float(self.value_max_var.get()),
            "replace_non_finite_with": 0.0,
        }
        zero_guard = {
            "enabled": _parse_bool(self.zero_guard_var.get()),
            "output_value": 0.0,
            "freq_enter_threshold": float(self.zero_enter_var.get()),
            "freq_exit_threshold": float(self.zero_exit_var.get()),
            "enter_consecutive_windows": int(self.zero_enter_count_var.get()),
            "exit_consecutive_windows": int(self.zero_exit_count_var.get()),
            "confidence_absz_threshold": 3.0,
        }
        postprocessing = {
            "enabled": _parse_bool(self.post_enabled_var.get()),
            "type": self.post_type_var.get().strip() or "none",
            "exp_smooth_alpha": float(self.smooth_alpha_var.get()),
            "kalman_q": float(self.kalman_q_var.get()),
            "kalman_r": float(self.kalman_r_var.get()),
            "apply_to_zero_guard": False,
            "reset_on_zero_guard": True,
        }
        uart = {
            "enabled": True,
            "uart_id": int(self.uart_id_var.get()),
            "tx_pin": int(self.tx_pin_var.get()),
            "rx_pin": int(self.rx_pin_var.get()),
            "baudrate": int(self.baudrate_var.get()),
            "bits": 8,
            "parity": "none",
            "stop": 1,
            "value_type": "int32",
            "byte_order": "big",
            "header": _parse_byte_list(self.header_var.get()),
            "tail": _parse_byte_list(self.tail_var.get()),
            "outer_frame_enabled": _parse_bool(self.outer_frame_var.get()),
            "outer_frame_count": int(self.outer_count_var.get()),
            "outer_header": _parse_byte_list(self.outer_header_var.get()),
            "outer_tail": _parse_byte_list(self.outer_tail_var.get()),
        }
        runtime_flags = {
            "mode": "uart_online",
            "quiet": _parse_bool(self.quiet_var.get()),
            "debug_predict_trace": _parse_bool(self.debug_trace_var.get()),
            "debug_tx_timing": _parse_bool(self.debug_tx_var.get()),
        }
        window_overrides = {
            "base_window_size": _parse_optional_int(self.base_window_var.get()),
            "base_step": _parse_optional_int(self.base_step_var.get()),
            "sequence_length": _parse_optional_int(self.seq_len_var.get()),
            "sequence_step": _parse_optional_int(self.seq_step_var.get()),
            "feature_mode": self.feature_mode_var.get().strip() or None,
        }
        return {
            "window_overrides": window_overrides,
            "value_guard": value_guard,
            "zero_guard": zero_guard,
            "postprocessing": postprocessing,
            "uart": uart,
            "runtime_flags": runtime_flags,
        }

    def generate_runtime_config(self) -> Path:
        if not self.selected_artifact:
            raise RuntimeError("请先选择一个 VQ 产物。")
        base_path = self._base_config_path()
        if not base_path.exists():
            raise FileNotFoundError("基础 runtime 配置不存在：{}".format(base_path))
        output_path = self._runtime_config_path()
        options = self._build_runtime_options()
        updated = build_runtime_from_vq_artifact(
            load_json(base_path),
            self.selected_artifact,
            append=bool(self.append_var.get()),
            model_index=int(self.model_index_var.get()),
            model_name=self.model_name_var.get().strip() or None,
            output_name=self.output_name_var.get().strip() or None,
            profile_name=self.profile_var.get().strip() or None,
            input_channels_text=self.input_channels_var.get().strip() or None,
            output_slots_text=self.output_slots_var.get().strip() or None,
            output_scale=_parse_optional_float(self.output_scale_var.get()),
            asset_subdir=self.asset_subdir_var.get().strip() or None,
            channel_count=_parse_optional_int(self.channel_count_var.get()),
            slot_count=_parse_optional_int(self.slot_count_var.get()),
            **options,
        )
        save_json(output_path, updated)
        self.output_config_var.set(_rel_or_abs(output_path))
        self.status_var.set("runtime 配置已生成：{}".format(output_path))
        self._log("runtime config generated: {}".format(output_path))
        return output_path

    def generate_and_build(self) -> None:
        try:
            config_path = self.generate_runtime_config()
        except Exception as exc:
            messagebox.showerror("生成配置失败", str(exc))
            return
        self.start_build(config_path)

    def start_build(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = self._runtime_config_path()
        if not config_path.exists():
            messagebox.showerror("配置不存在", "找不到配置文件：\n{}".format(config_path))
            return

        self.last_package_root = None
        self.last_zip_path = None
        self.open_folder_button.configure(state="disabled")
        self.open_zip_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.status_var.set("正在生成，请稍候...")
        self._clear_log()

        worker = threading.Thread(
            target=self._build_worker,
            args=(config_path, bool(self.make_zip_var.get())),
            daemon=True,
        )
        worker.start()
        self.notebook.select(3)

    def _build_worker(self, config_path: Path, make_zip: bool) -> None:
        try:
            rel_config = _rel_or_abs(config_path)
            self._log("配置文件：{}".format(rel_config))
            self._run_step("校验运行配置", [sys.executable, "scripts/validate_runtime_config.py", "--config", rel_config])
            output = self._run_step("生成部署包", [sys.executable, "scripts/make_deploy_package.py", "--config", rel_config])
            package_root = self._parse_package_root(output)
            manifest = package_root / "raw_cnn_k230" / "DEPLOY_MANIFEST.json"
            self._run_step(
                "校验部署包",
                [sys.executable, "scripts/verify_deploy_package.py", "--manifest", _rel_or_abs(manifest)],
            )
            zip_path = None
            if make_zip:
                zip_path = package_root.with_suffix(".zip")
                self._zip_package(package_root, zip_path)
            self.message_queue.put(("done", (package_root, zip_path)))
        except Exception as exc:
            self.message_queue.put(("error", str(exc)))

    def _run_step(self, title: str, command: list[str]) -> str:
        self._log("")
        self._log("== {} ==".format(title))
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
        if output.strip():
            self._log(output.rstrip())
        if proc.returncode != 0:
            raise RuntimeError("{}失败。".format(title))
        return output

    def _parse_package_root(self, output: str) -> Path:
        match = re.search(r"deploy package generated:\s*(.+)", output)
        if not match:
            raise RuntimeError("未能从打包输出中识别部署目录。")
        package_root = Path(match.group(1).strip())
        if not package_root.is_absolute():
            package_root = ROOT / package_root
        if not package_root.exists():
            raise RuntimeError("部署目录不存在：{}".format(package_root))
        return package_root

    def _zip_package(self, package_root: Path, zip_path: Path) -> None:
        self._log("")
        self._log("== 生成 zip ==")
        if zip_path.exists():
            raise RuntimeError("zip 已存在：{}".format(zip_path))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in package_root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(package_root))
        self._log("zip: {}".format(zip_path))

    def _set_text(self, text_widget, text: str) -> None:
        text_widget.configure(state="normal")
        text_widget.delete("1.0", END)
        text_widget.insert(END, text)
        text_widget.configure(state="disabled")

    def _log(self, text: str) -> None:
        if hasattr(self, "log_text"):
            self.message_queue.put(("log", text))

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", END)
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.configure(state="disabled")

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    package_root, zip_path = payload  # type: ignore[misc]
                    self.last_package_root = package_root
                    self.last_zip_path = zip_path
                    self.start_button.configure(state="normal")
                    self.open_folder_button.configure(state="normal")
                    if zip_path is not None:
                        self.open_zip_button.configure(state="normal")
                    self.status_var.set("生成完成：{}".format(package_root))
                    messagebox.showinfo("生成完成", "部署包生成完成。\n\n{}".format(package_root))
                elif kind == "error":
                    self.start_button.configure(state="normal")
                    self.status_var.set("生成失败。")
                    messagebox.showerror("生成失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def open_output_folder(self) -> None:
        if self.last_package_root and self.last_package_root.exists():
            os.startfile(str(self.last_package_root))  # type: ignore[attr-defined]

    def open_zip_folder(self) -> None:
        if self.last_zip_path and self.last_zip_path.exists():
            os.startfile(str(self.last_zip_path.parent))  # type: ignore[attr-defined]


def main() -> None:
    root = Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except Exception:
        pass
    VqDeployPackagerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
