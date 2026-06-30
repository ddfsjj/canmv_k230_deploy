"""生成 K230 板端部署包。

输出目录包含 SD 卡根目录启动器和 raw_cnn_k230 应用目录，可整体拷贝到 /sdcard。
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K230_DIR = ROOT / "raw_cnn_k230"
DEPLOY_ROOT = ROOT / "deploy_pkg" / "raw_cnn_k230"
if str(K230_DIR) not in sys.path:
    sys.path.insert(0, str(K230_DIR))

from runtime.config import load_runtime_config, to_legacy_multi_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build deploy_pkg from runtime.json.")
    parser.add_argument(
        "--config",
        default="raw_cnn_k230/configs/runtime.json",
        help="Runtime config path, relative to repo root by default.",
    )
    parser.add_argument(
        "--output",
        default="deploy_pkg/raw_cnn_k230",
        help="Output package directory, relative to repo root by default.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output package directory before rebuilding it.",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def write_text_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(rel, path):
    path = Path(path)
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_root_boot_py():
    return '''"""SD 卡根目录 boot.py：只负责把应用目录加入模块搜索路径。"""

import sys

APP_DIR = "/sdcard/raw_cnn_k230"

# 中文注释：根目录启动器保持很薄，业务代码统一放在 raw_cnn_k230 目录。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
'''


def build_root_main_py():
    return '''"""SD 卡根目录 main.py：上电后启动 raw_cnn_k230 统一 runtime。"""

import sys
import time
try:
    import uos as os  # type: ignore
except ImportError:
    import os

APP_DIR = "/sdcard/raw_cnn_k230"
RUNTIME_CONFIG_PATH = "/sdcard/raw_cnn_k230/configs/runtime.json"

# 中文注释：先把应用目录放到搜索路径最前面，再导入业务入口。
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def run_once():
    print("SD root launcher: raw_cnn_k230 unified runtime")
    print("runtime_config:", RUNTIME_CONFIG_PATH)
    import run_k230_infer as infer_app
    infer_app.OVERRIDE_CONFIG_PATH = RUNTIME_CONFIG_PATH
    infer_app.main()


def print_startup_error(exc):
    print("SD root auto-start error:", exc)
    print("APP_DIR:", APP_DIR)
    print("RUNTIME_CONFIG_PATH:", RUNTIME_CONFIG_PATH)
    try:
        print("cwd:", os.getcwd())
    except Exception as cwd_exc:
        print("cwd unavailable:", cwd_exc)


while True:
    try:
        run_once()
    except Exception as exc:
        # 中文注释：上电自启入口不直接退出，避免一次异常后板端停在空状态。
        print_startup_error(exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
'''


def collect_model_assets(legacy_cfg):
    assets = []
    seen = set()
    for model in legacy_cfg.get("models", []):
        paths = model.get("paths", {})
        for key in ("kmodel", "scaler_json"):
            rel = paths.get(key)
            if not rel:
                raise RuntimeError(f"model {model.get('name', '')} missing paths.{key}")
            src = K230_DIR / rel
            if not src.exists():
                raise FileNotFoundError(f"missing {key}: {src}")
            norm = rel.replace("\\", "/")
            if norm not in seen:
                seen.add(norm)
                assets.append((key, norm, src))
    return assets


def clean_deploy_output(output_dir, sdcard_root):
    """清理部署输出；默认 deploy_pkg 要按 SD 卡根目录整体重建。"""
    default_sdcard_root = (ROOT / "deploy_pkg").resolve()
    resolved_sdcard_root = sdcard_root.resolve()
    if resolved_sdcard_root == default_sdcard_root:
        if sdcard_root.exists():
            shutil.rmtree(sdcard_root)
        return

    # 中文注释：自定义输出路径可能指向用户目录，只清理本应用目录和生成的根启动文件。
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for rel in ("boot.py", "main.py"):
        path = sdcard_root / rel
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    output_dir = resolve_repo_path(args.output)
    sdcard_root = output_dir.parent
    cfg = load_runtime_config(str(config_path))
    legacy_cfg = to_legacy_multi_config(cfg)

    if args.clean:
        clean_deploy_output(output_dir, sdcard_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_files = []
    write_text_file(sdcard_root / "boot.py", build_root_boot_py())
    root_files.append("boot.py")
    write_text_file(sdcard_root / "main.py", build_root_main_py())
    root_files.append("main.py")
    root_file_records = []
    for rel in root_files:
        root_file_records.append(file_record(rel, sdcard_root / rel))

    copied = []
    for rel in ("boot.py", "main.py", "run_k230_infer.py", "run_k230_multi_infer.py"):
        src = K230_DIR / rel
        dst = output_dir / rel
        copy_file(src, dst)
        copied.append(rel)

    copy_tree(K230_DIR / "runtime", output_dir / "runtime")
    copied.append("runtime/")

    config_dst = output_dir / "configs" / "runtime.json"
    copy_file(config_path, config_dst)
    copied.append("configs/runtime.json")
    config_record = file_record("raw_cnn_k230/configs/runtime.json", config_dst)

    model_assets = collect_model_assets(legacy_cfg)
    manifest_assets = []
    for key, rel, src in model_assets:
        dst = output_dir / rel
        copy_file(src, dst)
        copied.append(rel)
        asset_record = file_record(rel, src)
        asset_record["kind"] = key
        manifest_assets.append(asset_record)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_name": legacy_cfg.get("name", ""),
        "source_config": config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
        "sdcard_root": sdcard_root.relative_to(ROOT).as_posix() if sdcard_root.is_relative_to(ROOT) else str(sdcard_root),
        "app_dir": output_dir.relative_to(ROOT).as_posix() if output_dir.is_relative_to(ROOT) else str(output_dir),
        "root_files": root_files,
        "root_file_records": root_file_records,
        "config": config_record,
        "files": copied,
        "assets": manifest_assets,
    }
    manifest_path = output_dir / "DEPLOY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("deploy package generated:", sdcard_root)
    print("app_dir:", output_dir)
    print("profile:", legacy_cfg.get("name", ""))
    print("root_files:", len(root_files))
    print("app_files:", len(copied))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
