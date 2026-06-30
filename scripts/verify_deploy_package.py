#!/usr/bin/env python3
"""校验部署包清单和实际文件是否一致。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_MANIFEST = Path("deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_record(base_dir: Path, record: dict, label: str) -> list[str]:
    errors: list[str] = []
    rel_path = Path(record["path"])
    path = base_dir / rel_path
    if not path.exists():
        return [f"{label} missing: {rel_path.as_posix()}"]
    expected_bytes = int(record.get("bytes", -1))
    if expected_bytes >= 0 and path.stat().st_size != expected_bytes:
        errors.append(
            f"{label} bytes mismatch: {rel_path.as_posix()} "
            f"manifest={expected_bytes} actual={path.stat().st_size}"
        )
    expected_sha = record.get("sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        errors.append(f"{label} sha256 mismatch: {rel_path.as_posix()}")
    return errors


def verify(manifest_path: Path) -> list[str]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    app_dir = manifest_path.parent
    sdcard_root = app_dir.parent
    errors: list[str] = []

    for record in manifest.get("root_file_records", []):
        errors.extend(check_record(sdcard_root, record, "root file"))

    config_record = manifest.get("config")
    if config_record:
        errors.extend(check_record(sdcard_root, config_record, "config"))

    for rel in manifest.get("files", []):
        path = app_dir / str(rel).rstrip("/")
        if not path.exists():
            errors.append(f"listed file missing: {rel}")
            continue
        if str(rel).endswith("/") and not path.is_dir():
            errors.append(f"listed path is not dir: {rel}")

    for record in manifest.get("assets", []):
        errors.extend(check_record(app_dir, record, "asset"))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify deploy package manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to DEPLOY_MANIFEST.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify(args.manifest)
    if errors:
        print("deploy package verification failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("deploy package verification ok")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
