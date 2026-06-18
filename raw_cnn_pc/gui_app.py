import json
import os
import subprocess
import sys
import time
import webbrowser
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ARTIFACT_ROOT = ROOT / "artifacts"
UI_RUN_ROOT = ARTIFACT_ROOT / "ui_runs"
UI_EXPORT_ROOT = ARTIFACT_ROOT / "ui_exports"
UI_COMPARE_ROOT = ARTIFACT_ROOT / "ui_compares"


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def rel_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_item(path: Path):
    return {
        "label": rel_label(path),
        "value": str(path.resolve()),
        "name": path.name,
    }


def scan_files():
    def files(patterns, roots):
        found = []
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                found.extend(root.rglob(pattern))
        return sorted({p.resolve() for p in found if p.is_file()}, key=lambda p: rel_label(p).lower())

    def config_files(root):
        if not root.exists():
            return []
        return sorted(root.rglob("*.json"), key=lambda p: rel_label(p).lower())

    def csv_dirs(roots):
        dirs = set()
        for root in roots:
            if not root.exists():
                continue
            for csv_file in root.rglob("*.csv"):
                if any(part in {"artifacts", "tmp", "__pycache__"} for part in csv_file.parts):
                    continue
                dirs.add(csv_file.parent.resolve())
        return sorted(dirs, key=lambda p: rel_label(p).lower())

    model_roots = [ROOT / "model", REPO_ROOT / "raw_cnn_k230" / "model"]
    kmodel_roots = [REPO_ROOT / "raw_cnn_k230" / "model", UI_EXPORT_ROOT]
    data_roots = [ROOT / "data", REPO_ROOT / "data", REPO_ROOT / "raw_cnn_k230" / "data"]

    return {
        "infer_configs": [file_item(p) for p in config_files(ROOT / "configs" / "infer")],
        "export_configs": [file_item(p) for p in config_files(ROOT / "configs" / "export")],
        "pth_files": [file_item(p) for p in files(["*.pth"], model_roots)],
        "scaler_files": [file_item(p) for p in files(["*.pkl"], model_roots)],
        "kmodel_files": [file_item(p) for p in files(["*.kmodel"], kmodel_roots)],
        "data_dirs": [
            {
                **file_item(p),
                "csv_count": len(list(p.glob("*.csv"))),
            }
            for p in csv_dirs(data_roots)
        ],
    }


def load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def run_command(args, cwd=ROOT, timeout=3600):
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )
    elapsed = time.perf_counter() - started
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "command": " ".join(str(x) for x in args),
        "output": proc.stdout,
    }


def timestamp_id(prefix):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def positive_int(value, fallback):
    if value in (None, ""):
        return fallback
    return int(value)


def optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def load_template(path_value, fallback):
    if path_value:
        return load_json_file(Path(path_value))
    return deepcopy(fallback)


def update_common_infer_config(cfg, payload):
    cfg.setdefault("data", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("model", {})
    cfg.setdefault("normalization", {})
    cfg.setdefault("runtime", {})

    cfg["data"]["test_data_dir"] = as_posix(Path(payload["data_dir"]))
    cfg["data"]["base_window_size"] = positive_int(payload.get("base_window_size"), cfg["data"].get("base_window_size", 500))
    cfg["data"]["base_step"] = positive_int(payload.get("base_step"), cfg["data"].get("base_step", 200))
    cfg["data"]["sequence_length"] = positive_int(payload.get("sequence_length"), cfg["data"].get("sequence_length", 5))
    cfg["data"]["sequence_step"] = positive_int(payload.get("sequence_step"), cfg["data"].get("sequence_step", 1))
    cfg["preprocessing"]["feature_mode"] = payload.get("feature_mode") or cfg["preprocessing"].get("feature_mode", "raw")
    cfg["preprocessing"]["filter_type"] = payload.get("filter_type") or cfg["preprocessing"].get("filter_type", "none")
    cfg["model"]["type"] = payload.get("model_type") or cfg["model"].get("type", "CNN-TCN")
    cfg["model"]["weights_path"] = as_posix(Path(payload["model_path"]))
    cfg["normalization"]["type"] = cfg["normalization"].get("type", "StandardScaler")
    cfg["normalization"]["scaler_path"] = as_posix(Path(payload["scaler_path"]))
    cfg["runtime"]["device"] = payload.get("device") or cfg["runtime"].get("device", "cpu")
    cfg["runtime"]["max_samples"] = optional_int(payload.get("max_samples"))
    return cfg


def update_export_config(cfg, payload, export_dir: Path):
    cfg.setdefault("paths", {})
    cfg.setdefault("data", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("model", {})
    cfg.setdefault("quantization", {})

    version = payload.get("version") or Path(payload["model_path"]).stem
    safe_version = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in version)
    paths = cfg["paths"]
    paths["weights_pth"] = as_posix(Path(payload["model_path"]))
    paths["scaler_pkl"] = as_posix(Path(payload["scaler_path"]))
    paths["onnx"] = as_posix(export_dir / f"{safe_version}.onnx")
    paths["kmodel"] = as_posix(export_dir / f"{safe_version}.kmodel")
    paths["scaler_json"] = as_posix(export_dir / f"{safe_version}_scaler.json")
    paths["calibration_npy"] = as_posix(export_dir / f"{safe_version}_calibration_input.npy")
    paths["calibration_data_dir"] = as_posix(Path(payload["calibration_data_dir"]))
    paths["test_data_dir"] = as_posix(Path(payload.get("test_data_dir") or payload["calibration_data_dir"]))
    paths["predictions_csv"] = as_posix(export_dir / f"{safe_version}_predictions.csv")
    paths["nncase_dump_dir"] = as_posix(export_dir / f"{safe_version}_nncase_dump")

    cfg["data"]["base_window_size"] = positive_int(payload.get("base_window_size"), cfg["data"].get("base_window_size", 500))
    cfg["data"]["base_step"] = positive_int(payload.get("base_step"), cfg["data"].get("base_step", 200))
    cfg["data"]["sequence_length"] = positive_int(payload.get("sequence_length"), cfg["data"].get("sequence_length", 5))
    cfg["data"]["sequence_step"] = positive_int(payload.get("sequence_step"), cfg["data"].get("sequence_step", 1))
    cfg["preprocessing"]["feature_mode"] = payload.get("feature_mode") or cfg["preprocessing"].get("feature_mode", "raw")
    cfg["preprocessing"]["filter_type"] = payload.get("filter_type") or cfg["preprocessing"].get("filter_type", "none")
    cfg["model"]["type"] = payload.get("model_type") or cfg["model"].get("type", "CNN-TCN")
    cfg["quantization"]["samples_count"] = optional_int(payload.get("samples_count"))
    cfg["quantization"]["sampling_strategy"] = payload.get("sampling_strategy") or cfg["quantization"].get("sampling_strategy", "first")
    cfg["quantization"]["random_seed"] = optional_int(payload.get("random_seed"))
    cfg["quantization"]["quant_type"] = payload.get("quant_type") or cfg["quantization"].get("quant_type", "uint8")
    cfg["quantization"]["weight_quant_type"] = payload.get("weight_quant_type") or cfg["quantization"].get("weight_quant_type", "uint8")
    cfg["quantization"]["calibrate_method"] = payload.get("calibrate_method") or cfg["quantization"].get("calibrate_method", "Kld")
    return cfg


def parse_metrics_from_log(text):
    metrics = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {
            "samples",
            "MAE",
            "RMSE",
            "prediction_csv",
            "total_samples",
            "pth_vs_kmodel_mae",
            "pth_vs_kmodel_rmse",
            "pth_vs_kmodel_max_abs",
            "kmodel_mae_vs_true",
            "kmodel_rmse_vs_true",
        }:
            metrics[key] = value.strip()
    return metrics


def api_infer(payload):
    run_dir = UI_RUN_ROOT / timestamp_id("infer")
    run_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_template(payload.get("infer_config"), {"name": "ui_infer"})
    cfg = update_common_infer_config(base_cfg, payload)
    cfg_path = run_dir / "infer_config.json"
    output_csv = run_dir / "predictions.csv"
    save_json_file(cfg_path, cfg)

    cmd = [sys.executable, "infer.py", "--config", as_posix(cfg_path), "--output", as_posix(output_csv)]
    if payload.get("max_samples"):
        cmd.extend(["--max_samples", str(int(payload["max_samples"]))])
    result = run_command(cmd)
    (run_dir / "run_log.txt").write_text(result["output"], encoding="utf-8")
    save_json_file(run_dir / "run_summary.json", {"config": cfg, "result": result, "output_csv": as_posix(output_csv)})
    return {
        **result,
        "run_dir": rel_label(run_dir),
        "output_csv": rel_label(output_csv),
        "metrics": parse_metrics_from_log(result["output"]),
    }


def api_export(payload):
    export_dir = UI_EXPORT_ROOT / timestamp_id("export")
    export_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_template(payload.get("export_config"), {"name": "ui_export", "paths": {}, "data": {}, "model": {}, "quantization": {}})
    cfg = update_export_config(base_cfg, payload, export_dir)
    cfg_path = export_dir / "export_config.json"
    save_json_file(cfg_path, cfg)

    cmd = [sys.executable, "build_kmodel.py", "--config", as_posix(cfg_path)]
    if payload.get("skip_compile"):
        cmd.append("--skip_compile")
    if payload.get("max_calib_samples"):
        cmd.extend(["--max_calib_samples", str(int(payload["max_calib_samples"]))])
    result = run_command(cmd, timeout=7200)
    (export_dir / "build_log.txt").write_text(result["output"], encoding="utf-8")
    save_json_file(export_dir / "export_summary.json", {"config": cfg, "result": result})
    return {
        **result,
        "export_dir": rel_label(export_dir),
        "config_path": rel_label(cfg_path),
        "onnx": rel_label(Path(cfg["paths"]["onnx"])),
        "kmodel": rel_label(Path(cfg["paths"]["kmodel"])),
        "scaler_json": rel_label(Path(cfg["paths"]["scaler_json"])),
        "calibration_npy": rel_label(Path(cfg["paths"]["calibration_npy"])),
    }


def api_compare(payload):
    compare_dir = UI_COMPARE_ROOT / timestamp_id("compare")
    compare_dir.mkdir(parents=True, exist_ok=True)
    infer_cfg = update_common_infer_config(load_template(payload.get("infer_config"), {"name": "ui_compare_infer"}), payload)
    export_cfg = load_template(payload.get("export_config"), {"name": "ui_compare_export", "paths": {}})
    export_cfg.setdefault("paths", {})
    export_cfg["paths"]["kmodel"] = as_posix(Path(payload["kmodel_path"]))

    infer_cfg_path = compare_dir / "infer_config.json"
    export_cfg_path = compare_dir / "export_config.json"
    summary_json = compare_dir / "summary.json"
    details_csv = compare_dir / "details.csv"
    per_csv_csv = compare_dir / "per_csv.csv"
    per_dryness_csv = compare_dir / "per_dryness.csv"
    save_json_file(infer_cfg_path, infer_cfg)
    save_json_file(export_cfg_path, export_cfg)

    cmd = [
        sys.executable,
        "compare_pth_kmodel.py",
        "--infer_config",
        as_posix(infer_cfg_path),
        "--export_config",
        as_posix(export_cfg_path),
        "--data_dir",
        as_posix(Path(payload["data_dir"])),
        "--summary_json",
        as_posix(summary_json),
        "--details_csv",
        as_posix(details_csv),
        "--per_csv_csv",
        as_posix(per_csv_csv),
        "--per_dryness_csv",
        as_posix(per_dryness_csv),
    ]
    for key in ("max_samples", "max_per_dryness", "log_every", "start_index", "end_index"):
        if payload.get(key) not in (None, ""):
            cmd.extend([f"--{key}", str(int(payload[key]))])

    result = run_command(cmd, timeout=7200)
    (compare_dir / "compare_log.txt").write_text(result["output"], encoding="utf-8")
    summary = load_json_file(summary_json) if summary_json.exists() else None
    save_json_file(compare_dir / "compare_run.json", {"infer_config": infer_cfg, "export_config": export_cfg, "result": result})
    return {
        **result,
        "compare_dir": rel_label(compare_dir),
        "summary_json": rel_label(summary_json),
        "details_csv": rel_label(details_csv),
        "per_csv_csv": rel_label(per_csv_csv),
        "per_dryness_csv": rel_label(per_dryness_csv),
        "summary": summary,
        "metrics": summary or parse_metrics_from_log(result["output"]),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Raw CNN K230 工作台</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d9e0e7;
      --text: #18212b;
      --muted: #657384;
      --primary: #1769aa;
      --primary-strong: #0f4f86;
      --ok: #16794c;
      --bad: #b42318;
      --warn: #a15c00;
      --shadow: 0 10px 28px rgba(20, 35, 50, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 28px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { font-size: 20px; margin: 0; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    main {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .sidebar, .content, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .sidebar { padding: 12px; height: fit-content; position: sticky; top: 78px; }
    .tab {
      width: 100%;
      border: 0;
      background: transparent;
      color: var(--text);
      text-align: left;
      padding: 11px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      margin-bottom: 4px;
    }
    .tab.active { background: #e9f2fb; color: var(--primary-strong); font-weight: 600; }
    .content { padding: 18px; min-height: 720px; }
    .section { display: none; }
    .section.active { display: block; }
    .section h2 { font-size: 18px; margin: 0 0 14px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 12px;
    }
    .field { grid-column: span 4; min-width: 0; }
    .field.wide { grid-column: span 8; }
    .field.full { grid-column: 1 / -1; }
    label { display: block; font-size: 12px; color: var(--muted); margin: 0 0 5px; }
    select, input {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      font-size: 13px;
      color: var(--text);
      background: #fff;
    }
    select:focus, input:focus { outline: 2px solid #b8d7f0; border-color: var(--primary); }
    .panel { padding: 14px; margin: 14px 0; box-shadow: none; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    button.primary, button.secondary {
      border: 1px solid var(--primary);
      border-radius: 6px;
      min-height: 36px;
      padding: 0 14px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { color: #fff; background: var(--primary); }
    button.primary:hover { background: var(--primary-strong); }
    button.secondary { color: var(--primary-strong); background: #fff; }
    .status {
      border-radius: 6px;
      padding: 10px 12px;
      background: #f7f9fb;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0;
    }
    .status.ok { border-color: #a9d8bf; color: var(--ok); background: #f1fbf5; }
    .status.bad { border-color: #f0b8b2; color: var(--bad); background: #fff5f4; }
    pre {
      background: #111820;
      color: #d7e1ea;
      padding: 14px;
      border-radius: 8px;
      overflow: auto;
      max-height: 360px;
      font-size: 12px;
      line-height: 1.45;
    }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
      min-height: 74px;
    }
    .metric .name { color: var(--muted); font-size: 12px; }
    .metric .value { margin-top: 8px; font-size: 18px; font-weight: 700; overflow-wrap: anywhere; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.55; }
    .path-list { font-size: 12px; color: var(--muted); line-height: 1.6; overflow-wrap: anywhere; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .field, .field.wide { grid-column: 1 / -1; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Raw CNN K230 工作台</h1>
      <div class="sub">PC 推理、KModel 生成、PTH vs KModel 对比。不会修改板端配置。</div>
    </div>
    <button class="secondary" onclick="scan()">重新扫描</button>
  </header>
  <main>
    <aside class="sidebar">
      <button class="tab active" data-tab="infer">PTH 预测</button>
      <button class="tab" data-tab="export">KModel 生成</button>
      <button class="tab" data-tab="compare">PTH vs KModel</button>
      <button class="tab" data-tab="scan">扫描结果</button>
    </aside>
    <section class="content">
      <div id="status" class="status">正在扫描项目文件...</div>

      <div id="infer" class="section active">
        <h2>PTH 预测</h2>
        <div class="hint">选择现有推理配置作为结构模板，再覆盖模型、标尺、数据和窗口参数。本页只生成本次运行记录，不写回正式配置。</div>
        <div class="panel grid">
          <div class="field wide"><label>推理配置模板</label><select id="infer_config"></select></div>
          <div class="field"><label>模型类型</label><select id="infer_model_type"></select></div>
          <div class="field wide"><label>PTH 模型</label><select id="infer_model"></select></div>
          <div class="field wide"><label>scaler.pkl</label><select id="infer_scaler"></select></div>
          <div class="field wide"><label>数据目录</label><select id="infer_data"></select></div>
          <div class="field"><label>base_window_size</label><input id="infer_base_window_size" type="number" value="500"></div>
          <div class="field"><label>base_step</label><input id="infer_base_step" type="number" value="200"></div>
          <div class="field"><label>sequence_length</label><input id="infer_sequence_length" type="number" value="5"></div>
          <div class="field"><label>sequence_step</label><input id="infer_sequence_step" type="number" value="1"></div>
          <div class="field"><label>feature_mode</label><select id="infer_feature_mode"></select></div>
          <div class="field"><label>filter_type</label><input id="infer_filter_type" value="none"></div>
          <div class="field"><label>max_samples</label><input id="infer_max_samples" type="number" placeholder="空=全量"></div>
        </div>
        <div class="actions"><button class="primary" onclick="runInfer()">运行 PTH 预测</button></div>
        <div id="infer_metrics" class="cards"></div>
        <pre id="infer_log"></pre>
      </div>

      <div id="export" class="section">
        <h2>KModel 生成</h2>
        <div class="hint">选择导出模板和量化参数，产物输出到 <code>raw_cnn_pc/artifacts/ui_exports</code>。不写入 <code>raw_cnn_k230</code>。</div>
        <div class="panel grid">
          <div class="field wide"><label>导出配置模板</label><select id="export_config"></select></div>
          <div class="field"><label>模型类型</label><select id="export_model_type"></select></div>
          <div class="field wide"><label>PTH 模型</label><select id="export_model"></select></div>
          <div class="field wide"><label>scaler.pkl</label><select id="export_scaler"></select></div>
          <div class="field wide"><label>校准数据目录</label><select id="export_data"></select></div>
          <div class="field"><label>版本名</label><input id="export_version" placeholder="默认使用 pth 文件名"></div>
          <div class="field"><label>base_window_size</label><input id="export_base_window_size" type="number" value="500"></div>
          <div class="field"><label>base_step</label><input id="export_base_step" type="number" value="200"></div>
          <div class="field"><label>sequence_length</label><input id="export_sequence_length" type="number" value="5"></div>
          <div class="field"><label>sequence_step</label><input id="export_sequence_step" type="number" value="1"></div>
          <div class="field"><label>feature_mode</label><select id="export_feature_mode"></select></div>
          <div class="field"><label>filter_type</label><input id="export_filter_type" value="none"></div>
          <div class="field"><label>samples_count</label><input id="export_samples_count" type="number" value="512"></div>
          <div class="field"><label>sampling_strategy</label><select id="export_sampling_strategy"></select></div>
          <div class="field"><label>quant_type</label><select id="export_quant_type"></select></div>
          <div class="field"><label>weight_quant_type</label><select id="export_weight_quant_type"></select></div>
          <div class="field"><label>calibrate_method</label><select id="export_calibrate_method"></select></div>
          <div class="field"><label>random_seed</label><input id="export_random_seed" type="number" value="20260414"></div>
          <div class="field"><label>max_calib_samples</label><input id="export_max_calib_samples" type="number" placeholder="空=按配置"></div>
          <div class="field"><label>跳过 nncase 编译</label><select id="export_skip_compile"><option value="">否，生成 KModel</option><option value="1">是，只导出 ONNX/scaler</option></select></div>
        </div>
        <div class="actions"><button class="primary" onclick="runExport()">生成 KModel</button></div>
        <div id="export_metrics" class="cards"></div>
        <pre id="export_log"></pre>
      </div>

      <div id="compare" class="section">
        <h2>PTH vs KModel 对比</h2>
        <div class="hint">可以选择历史 KModel，也可以先生成 KModel 后点击重新扫描再选择。对比结果输出到 <code>raw_cnn_pc/artifacts/ui_compares</code>。</div>
        <div class="panel grid">
          <div class="field wide"><label>推理配置模板</label><select id="compare_infer_config"></select></div>
          <div class="field wide"><label>导出配置模板</label><select id="compare_export_config"></select></div>
          <div class="field"><label>模型类型</label><select id="compare_model_type"></select></div>
          <div class="field wide"><label>PTH 模型</label><select id="compare_model"></select></div>
          <div class="field wide"><label>scaler.pkl</label><select id="compare_scaler"></select></div>
          <div class="field wide"><label>KModel</label><select id="compare_kmodel"></select></div>
          <div class="field wide"><label>数据目录</label><select id="compare_data"></select></div>
          <div class="field"><label>base_window_size</label><input id="compare_base_window_size" type="number" value="500"></div>
          <div class="field"><label>base_step</label><input id="compare_base_step" type="number" value="200"></div>
          <div class="field"><label>sequence_length</label><input id="compare_sequence_length" type="number" value="5"></div>
          <div class="field"><label>sequence_step</label><input id="compare_sequence_step" type="number" value="1"></div>
          <div class="field"><label>feature_mode</label><select id="compare_feature_mode"></select></div>
          <div class="field"><label>filter_type</label><input id="compare_filter_type" value="none"></div>
          <div class="field"><label>max_samples</label><input id="compare_max_samples" type="number" placeholder="空=全量"></div>
          <div class="field"><label>max_per_dryness</label><input id="compare_max_per_dryness" type="number" placeholder="空=不限制"></div>
          <div class="field"><label>start_index</label><input id="compare_start_index" type="number" value="0"></div>
          <div class="field"><label>end_index</label><input id="compare_end_index" type="number" placeholder="空=末尾"></div>
          <div class="field"><label>log_every</label><input id="compare_log_every" type="number" value="500"></div>
        </div>
        <div class="actions"><button class="primary" onclick="runCompare()">运行对比</button></div>
        <div id="compare_metrics" class="cards"></div>
        <pre id="compare_log"></pre>
      </div>

      <div id="scan" class="section">
        <h2>扫描结果</h2>
        <div class="panel">
          <div class="path-list" id="scan_result"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const modelTypes = ["CNN-All", "CNN-LSTM", "CNN-TCN", "cnn_tcn_seg3_soft_stats_moe"];
    const featureModes = ["raw", "window_demean", "window_rel_demean"];
    const samplingStrategies = ["first", "random", "per_dryness_uniform", "high_dryness_weighted"];
    const quantTypes = ["uint8", "int8", "int16"];
    const weightQuantTypes = ["uint8", "int8", "int16"];
    const calibrateMethods = ["Kld", "NoClip", "MinMax"];
    let scanData = null;

    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
        document.querySelectorAll(".section").forEach(x => x.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
      });
    });

    function status(text, kind="") {
      const el = document.getElementById("status");
      el.className = "status " + kind;
      el.textContent = text;
    }

    function fillOptions(id, items, emptyText) {
      const el = document.getElementById(id);
      el.innerHTML = "";
      if (emptyText) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = emptyText;
        el.appendChild(opt);
      }
      items.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item.value ?? item;
        opt.textContent = item.label ?? item;
        el.appendChild(opt);
      });
    }

    function fillStatic() {
      ["infer_model_type", "export_model_type", "compare_model_type"].forEach(id => fillOptions(id, modelTypes));
      ["infer_feature_mode", "export_feature_mode", "compare_feature_mode"].forEach(id => fillOptions(id, featureModes));
      fillOptions("export_sampling_strategy", samplingStrategies);
      fillOptions("export_quant_type", quantTypes);
      fillOptions("export_weight_quant_type", weightQuantTypes);
      fillOptions("export_calibrate_method", calibrateMethods);
    }

    async function scan() {
      status("正在扫描项目文件...");
      const res = await fetch("/api/scan");
      scanData = await res.json();
      fillOptions("infer_config", scanData.infer_configs);
      fillOptions("compare_infer_config", scanData.infer_configs);
      fillOptions("export_config", scanData.export_configs);
      fillOptions("compare_export_config", scanData.export_configs, "可空");
      ["infer_model", "export_model", "compare_model"].forEach(id => fillOptions(id, scanData.pth_files));
      ["infer_scaler", "export_scaler", "compare_scaler"].forEach(id => fillOptions(id, scanData.scaler_files));
      ["infer_data", "export_data", "compare_data"].forEach(id => fillOptions(id, scanData.data_dirs.map(x => ({label: `${x.label} (${x.csv_count} CSV)`, value: x.value}))));
      fillOptions("compare_kmodel", scanData.kmodel_files);
      renderScan();
      status(`扫描完成：${scanData.pth_files.length} 个 PTH，${scanData.scaler_files.length} 个 scaler，${scanData.kmodel_files.length} 个 KModel。`, "ok");
    }

    function renderScan() {
      const parts = [
        ["PTH", scanData.pth_files],
        ["scaler.pkl", scanData.scaler_files],
        ["KModel", scanData.kmodel_files],
        ["推理配置", scanData.infer_configs],
        ["导出配置", scanData.export_configs],
        ["数据目录", scanData.data_dirs],
      ];
      document.getElementById("scan_result").innerHTML = parts.map(([title, items]) => {
        const rows = items.map(x => `&nbsp;&nbsp;${x.label}${x.csv_count ? ` (${x.csv_count} CSV)` : ""}`).join("<br>");
        return `<strong>${title} (${items.length})</strong><br>${rows || "&nbsp;&nbsp;无"}`;
      }).join("<br><br>");
    }

    function payload(prefix) {
      const get = id => document.getElementById(`${prefix}_${id}`)?.value ?? "";
      return {
        infer_config: get("config"),
        export_config: get("config"),
        model_type: get("model_type"),
        model_path: get("model"),
        scaler_path: get("scaler"),
        data_dir: get("data"),
        calibration_data_dir: get("data"),
        base_window_size: get("base_window_size"),
        base_step: get("base_step"),
        sequence_length: get("sequence_length"),
        sequence_step: get("sequence_step"),
        feature_mode: get("feature_mode"),
        filter_type: get("filter_type"),
        max_samples: get("max_samples"),
      };
    }

    function renderMetrics(id, metrics) {
      const el = document.getElementById(id);
      if (!metrics) { el.innerHTML = ""; return; }
      const entries = Object.entries(metrics).slice(0, 12);
      el.innerHTML = entries.map(([k, v]) => `<div class="metric"><div class="name">${k}</div><div class="value">${v}</div></div>`).join("");
    }

    async function postJson(url, body) {
      const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "请求失败");
      return data;
    }

    async function runInfer() {
      status("正在运行 PTH 预测...");
      document.getElementById("infer_log").textContent = "";
      try {
        const body = payload("infer");
        const data = await postJson("/api/infer", body);
        document.getElementById("infer_log").textContent = data.output;
        renderMetrics("infer_metrics", {...data.metrics, run_dir: data.run_dir, output_csv: data.output_csv});
        status(data.returncode === 0 ? "PTH 预测完成。" : "PTH 预测失败，请看日志。", data.returncode === 0 ? "ok" : "bad");
      } catch (err) {
        status(err.message, "bad");
      }
    }

    async function runExport() {
      status("正在生成 KModel...");
      document.getElementById("export_log").textContent = "";
      try {
        const body = payload("export");
        body.export_config = document.getElementById("export_config").value;
        body.version = document.getElementById("export_version").value;
        body.samples_count = document.getElementById("export_samples_count").value;
        body.sampling_strategy = document.getElementById("export_sampling_strategy").value;
        body.quant_type = document.getElementById("export_quant_type").value;
        body.weight_quant_type = document.getElementById("export_weight_quant_type").value;
        body.calibrate_method = document.getElementById("export_calibrate_method").value;
        body.random_seed = document.getElementById("export_random_seed").value;
        body.max_calib_samples = document.getElementById("export_max_calib_samples").value;
        body.skip_compile = Boolean(document.getElementById("export_skip_compile").value);
        const data = await postJson("/api/export", body);
        document.getElementById("export_log").textContent = data.output;
        renderMetrics("export_metrics", {export_dir: data.export_dir, kmodel: data.kmodel, onnx: data.onnx, scaler_json: data.scaler_json});
        status(data.returncode === 0 ? "KModel 生成流程完成。" : "KModel 生成失败，请看日志。", data.returncode === 0 ? "ok" : "bad");
        await scan();
      } catch (err) {
        status(err.message, "bad");
      }
    }

    async function runCompare() {
      status("正在运行 PTH vs KModel 对比...");
      document.getElementById("compare_log").textContent = "";
      try {
        const body = payload("compare");
        body.infer_config = document.getElementById("compare_infer_config").value;
        body.export_config = document.getElementById("compare_export_config").value;
        body.kmodel_path = document.getElementById("compare_kmodel").value;
        body.max_per_dryness = document.getElementById("compare_max_per_dryness").value;
        body.start_index = document.getElementById("compare_start_index").value;
        body.end_index = document.getElementById("compare_end_index").value;
        body.log_every = document.getElementById("compare_log_every").value;
        const data = await postJson("/api/compare", body);
        document.getElementById("compare_log").textContent = data.output;
        renderMetrics("compare_metrics", {...data.metrics, compare_dir: data.compare_dir});
        status(data.returncode === 0 ? "对比完成。" : "对比失败，请看日志。", data.returncode === 0 ? "ok" : "bad");
      } catch (err) {
        status(err.message, "bad");
      }
    }

    fillStatic();
    scan();
  </script>
</body>
</html>"""


class GuiHandler(BaseHTTPRequestHandler):
    def _send(self, status_code, content, content_type="application/json; charset=utf-8"):
        if isinstance(content, (dict, list)):
            body = json.dumps(content, ensure_ascii=False).encode("utf-8")
        elif isinstance(content, str):
            body = content.encode("utf-8")
        else:
            body = content
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, HTML, "text/html; charset=utf-8")
            elif parsed.path == "/api/scan":
                self._send(200, scan_files())
            elif parsed.path == "/api/config":
                qs = parse_qs(parsed.query)
                path_value = qs.get("path", [""])[0]
                if not path_value:
                    self._send(400, {"error": "missing path"})
                    return
                path = Path(path_value)
                self._send(200, load_json_file(path))
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/infer":
                self._send(200, api_infer(payload))
            elif parsed.path == "/api/export":
                self._send(200, api_export(payload))
            elif parsed.path == "/api/compare":
                self._send(200, api_compare(payload))
            else:
                self._send(404, {"error": "not found"})
        except subprocess.TimeoutExpired as exc:
            self._send(500, {"error": f"命令超时: {exc}"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print("[gui]", fmt % args)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Raw CNN K230 local web GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), GuiHandler)
    url = f"http://{args.host}:{args.port}/"
    print("Raw CNN K230 工作台:", url)
    print("页面不会修改 raw_cnn_k230 板端配置。按 Ctrl+C 停止。")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
