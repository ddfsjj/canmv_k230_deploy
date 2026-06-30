"""统一 K230 runtime 入口。"""

try:
    import sys  # type: ignore
except ImportError:
    sys = None  # type: ignore

try:
    import uos as os  # type: ignore
except ImportError:
    import os  # type: ignore

from runtime.config import DEFAULT_RUNTIME_CONFIG_PATH, load_runtime_config, to_legacy_multi_config
from runtime import uart as runtime_uart


OVERRIDE_CONFIG_PATH = None


def _norm_path(path):
    return str(path).replace("\\", "/")


def _join_path(root, path):
    text = str(path)
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        return _norm_path(text)
    return _norm_path(root.rstrip("/\\") + "/" + text)


def _safe_getcwd():
    try:
        return _norm_path(os.getcwd())
    except Exception as exc:
        return "<cwd unavailable: {}>".format(exc)


def _print_startup_context(root, config_path):
    print("runtime_root:", root)
    print("runtime_cwd:", _safe_getcwd())
    print("runtime_config_resolved:", config_path)
    if sys is not None:
        try:
            print("runtime_sys_path_0:", sys.path[0] if sys.path else "")
        except Exception:
            pass


def _assert_file_exists(label, path):
    try:
        os.stat(path)
    except Exception as exc:
        raise RuntimeError("{} missing: {} ({})".format(label, path, exc))


def _validate_runtime_files(root, legacy_cfg):
    # 中文注释：板端启动时做轻量文件存在性校验，缺文件时给出明确路径。
    for idx, model in enumerate(legacy_cfg.get("models", [])):
        paths = model.get("paths", {})
        model_name = model.get("name", "model_{}".format(idx))
        kmodel = paths.get("kmodel", "")
        scaler_json = paths.get("scaler_json", "")
        _assert_file_exists(
            "models[{}] {} kmodel".format(idx, model_name),
            _join_path(root, kmodel),
        )
        _assert_file_exists(
            "models[{}] {} scaler_json".format(idx, model_name),
            _join_path(root, scaler_json),
        )


def detect_root():
    """中文注释：优先查找新版 runtime.json，再兼容旧配置目录。"""
    candidates = []
    try:
        candidates.append(_norm_path(os.getcwd()))
    except Exception:
        pass
    here = globals().get("__file__", "")
    if here:
        here = _norm_path(here)
        if "/" in here:
            candidates.append(here.rsplit("/", 2)[0])
    candidates.append("/sdcard/raw_cnn_k230")
    candidates.append("/sdcard")

    seen = set()
    for root in candidates:
        if not root or root in seen:
            continue
        seen.add(root)
        try:
            os.stat(_join_path(root, DEFAULT_RUNTIME_CONFIG_PATH))
            return root
        except Exception:
            pass
        try:
            os.stat(_join_path(root, "configs/k230_config_multi.json"))
            return root
        except Exception:
            pass
    return candidates[0] if candidates else "."


def resolve_config_path(root, cli_args):
    selected = None
    args = list(cli_args or [])
    idx = 0
    while idx < len(args):
        token = str(args[idx])
        if token == "--config":
            if idx + 1 >= len(args):
                raise ValueError("--config requires a path argument.")
            selected = str(args[idx + 1])
            break
        if token.lower().endswith(".json"):
            selected = token
            break
        idx += 1
    if OVERRIDE_CONFIG_PATH:
        selected = str(OVERRIDE_CONFIG_PATH)
    if not selected:
        selected = DEFAULT_RUNTIME_CONFIG_PATH
    if selected.startswith("/") or (len(selected) > 1 and selected[1] == ":"):
        return _norm_path(selected)
    return _join_path(root, selected)


def main(config_path=None):
    """中文注释：唯一运行入口，新配置和旧配置都统一转到同一个多模型后端。"""
    root = detect_root()
    cli_args = []
    if sys is not None:
        try:
            cli_args = list(sys.argv[1:])
        except Exception:
            cli_args = []
    if config_path is None:
        config_path = resolve_config_path(root, cli_args)
    elif not (str(config_path).startswith("/") or (len(str(config_path)) > 1 and str(config_path)[1] == ":")):
        config_path = _join_path(root, config_path)

    cfg = load_runtime_config(config_path)
    legacy_cfg = to_legacy_multi_config(cfg)
    _validate_runtime_files(root, legacy_cfg)
    runtime_cfg = legacy_cfg.get("runtime", {})
    mode = str(runtime_cfg.get("mode", "uart_online")).strip().lower()

    import run_k230_infer as base

    if mode == "uart":
        mode = "uart_online"
        legacy_cfg["runtime"]["mode"] = mode
    if mode == "csv":
        mode = "csv_cached"
        legacy_cfg["runtime"]["mode"] = mode

    print("=== K230 Unified Runtime ===")
    _print_startup_context(root, config_path)
    print("config_path:", config_path)
    print("config_name:", legacy_cfg.get("name", ""))
    print("mode:", mode)
    print("model_bindings:", len(legacy_cfg.get("models", [])))

    if mode == "uart_online":
        from runtime import online as runtime_online

        uart_sender = runtime_uart.UartDrynessSender(legacy_cfg.get("uart", {}))
        runtime_online.run_uart_online(
            cfg=legacy_cfg,
            root=root,
            uart_sender=uart_sender,
        )
        return

    if mode == "csv_cached":
        from runtime import csv as runtime_csv

        runtime_csv.run_csv_cached(
            cfg=legacy_cfg,
            root=root,
        )
        return

    raise ValueError("Unsupported unified runtime mode: {}".format(mode))
