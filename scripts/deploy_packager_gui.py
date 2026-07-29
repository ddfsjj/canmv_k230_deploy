"""GUI launcher for building K230 deploy packages."""

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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "raw_cnn_k230" / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "runtime.json"


class DeployPackagerGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("K230 部署包生成器")
        self.root.geometry("820x560")
        self.root.minsize(760, 500)

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.last_package_root: Path | None = None
        self.last_zip_path: Path | None = None

        self.config_var = StringVar(value=self._default_config_text())
        self.make_zip_var = BooleanVar(value=True)
        self.status_var = StringVar(value="请选择配置文件，然后开始生成。")

        self._build_ui()
        self._poll_messages()

    def _default_config_text(self) -> str:
        if DEFAULT_CONFIG.exists():
            return str(DEFAULT_CONFIG.relative_to(ROOT))
        fallback = CONFIG_DIR / "runtime.json"
        if fallback.exists():
            return str(fallback.relative_to(ROOT))
        return ""

    def _config_choices(self) -> list[str]:
        if not CONFIG_DIR.exists():
            return []
        configs = sorted(CONFIG_DIR.glob("runtime*.json"))
        return [str(path.relative_to(ROOT)) for path in configs]

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        title = ttk.Label(outer, text="K230 部署包生成器", font=("Microsoft YaHei UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")

        config_frame = ttk.LabelFrame(outer, text="运行配置", padding=12)
        config_frame.grid(row=1, column=0, sticky="ew", pady=(14, 10))
        config_frame.columnconfigure(0, weight=1)

        self.config_combo = ttk.Combobox(
            config_frame,
            textvariable=self.config_var,
            values=self._config_choices(),
        )
        self.config_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(config_frame, text="选择...", command=self.choose_config).grid(row=0, column=1)

        options_frame = ttk.Frame(outer)
        options_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        options_frame.columnconfigure(4, weight=1)

        ttk.Checkbutton(options_frame, text="生成 zip", variable=self.make_zip_var).grid(
            row=0, column=0, sticky="w", padx=(0, 16)
        )
        self.start_button = ttk.Button(options_frame, text="开始生成", command=self.start_build)
        self.start_button.grid(row=0, column=1, padx=(0, 8))
        self.open_folder_button = ttk.Button(
            options_frame,
            text="打开输出目录",
            command=self.open_output_folder,
            state="disabled",
        )
        self.open_folder_button.grid(row=0, column=2, padx=(0, 8))
        self.open_zip_button = ttk.Button(
            options_frame,
            text="打开 zip 位置",
            command=self.open_zip_folder,
            state="disabled",
        )
        self.open_zip_button.grid(row=0, column=3)

        log_frame = ttk.LabelFrame(outer, text="执行记录", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = self._make_log_text(log_frame)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        status = ttk.Label(outer, textvariable=self.status_var)
        status.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _make_log_text(self, parent):
        from tkinter import Text

        text = Text(parent, height=18, wrap="word", font=("Consolas", 10))
        text.configure(state="disabled")
        return text

    def choose_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 runtime 配置",
            initialdir=str(CONFIG_DIR),
            filetypes=[("JSON 配置", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            path = Path(selected)
            try:
                self.config_var.set(str(path.relative_to(ROOT)))
            except ValueError:
                self.config_var.set(str(path))

    def start_build(self) -> None:
        config_text = self.config_var.get().strip().strip('"')
        if not config_text:
            messagebox.showwarning("缺少配置", "请先选择一个 runtime 配置文件。")
            return

        config_path = Path(config_text)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
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

    def _build_worker(self, config_path: Path, make_zip: bool) -> None:
        try:
            rel_config = self._display_path(config_path)
            self._log("配置文件：{}".format(rel_config))
            self._run_step("校验运行配置", [sys.executable, "scripts/validate_runtime_config.py", "--config", rel_config])

            output = self._run_step(
                "生成部署包",
                [sys.executable, "scripts/make_deploy_package.py", "--config", rel_config],
            )
            package_root = self._parse_package_root(output)
            manifest = package_root / "raw_cnn_k230" / "DEPLOY_MANIFEST.json"
            self._run_step(
                "校验部署包",
                [sys.executable, "scripts/verify_deploy_package.py", "--manifest", self._display_path(manifest)],
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

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    def _log(self, text: str) -> None:
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
    DeployPackagerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
