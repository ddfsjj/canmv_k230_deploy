import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
import torch.nn as nn

import compare_pth_kmodel
import infer

"""
澶у皢鍐涳紝杩欎釜鑴氭湰鐢ㄤ簬鍋氣€滃垎灞傜骇鐨?PTH vs ONNX 瀵规瘮鈥濄€?
瀹冭В鍐崇殑闂涓嶆槸鐩存帴璇诲彇 kmodel 鐨勬瘡灞傝緭鍑猴紝鑰屾槸鍏堝洖绛斾竴涓洿鍩虹鐨勯棶棰橈細
1. 鍚屼竴涓牱鏈粠 `.pth` 璧板埌 `onnx` 鏃讹紝鍝竴灞傚紑濮嬫紓锛?2. 婕傜Щ鏄嚭鍦ㄥ嵎绉€佹睜鍖栥€丩STM 杩樻槸鏈€缁堝叏杩炴帴锛?
涓轰粈涔堝厛鍋氳繖涓細
1. 褰撳墠 nncase 鐨?Python Simulator 鍙兘鐩存帴鎷挎渶缁堣緭鍑猴紝涓嶈兘鐩存帴鍍?PyTorch hook 涓€鏍烽€愬眰鍙栦腑闂村紶閲忋€?2. 浣嗗鏋滆繛 `.pth -> onnx` 杩欎竴娈甸兘宸茬粡寮€濮嬫紓锛岄偅闂杩樻病璧板埌 kmodel 灏卞凡缁忓嚭鐜颁簡銆?3. 濡傛灉 `.pth -> onnx` 鍩烘湰涓€鑷达紝鑰屾渶缁?`kmodel` 宸緢澶э紝閭ｉ棶棰樺氨鏇存帴杩戦噺鍖?/ nncase 缂栬瘧闃舵銆?
鑴氭湰鍋氱殑浜嬫儏锛?1. 浠?CSV 鏁版嵁鐩綍閲屾瀯寤哄拰姝ｅ紡鎺ㄧ悊涓€鑷寸殑鏍锋湰銆?2. 閫変竴涓牱鏈紝鍋氬悓鏍风殑 StandardScaler 鏍囧噯鍖栥€?3. 鐢?PyTorch 璋冭瘯鍖呰鍣ㄨ緭鍑哄叧閿腑闂村眰寮犻噺銆?4. 瀵煎嚭涓€涓€滃杈撳嚭璋冭瘯 ONNX鈥濓紝璁╄繖浜涘叧閿紶閲忎篃浣滀负 ONNX 杈撳嚭銆?5. 鐢?ONNX ReferenceEvaluator 璺戝悓涓€涓牱鏈紝閫愬眰瀵规瘮璇樊銆?
杈撳嚭缁撴灉锛?1. `layerwise_summary.json`锛氭€讳綋淇℃伅鍜屾瘡灞傝宸憳瑕?2. `layerwise_metrics.csv`锛氶€愬眰褰㈢姸銆丮AE銆丷MSE銆佹渶澶х粷瀵硅宸?3. `pth_tensors.npz`锛歅yTorch 姣忓眰寮犻噺
4. `onnx_tensors.npz`锛歄NNX 姣忓眰寮犻噺
5. `debug_model.onnx`锛氱敤浜庡垎灞傚姣旂殑澶氳緭鍑?ONNX
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Compare layerwise outputs between PTH and debug ONNX.")
    parser.add_argument("--infer_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="layerwise_compare")
    parser.add_argument("--sample_index", type=int, default=0, help="全局样本索引，从构建后的样本序列里选择。")
    parser.add_argument("--csv_name", type=str, default=None, help="可选，只从指定 CSV 对应的样本中选择。")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class DebugCNNLSTM(nn.Module):
    """把 CNN-LSTM 的关键中间张量全部显式返回。"""

    stage_names = [
        "input",
        "reshape_in",
        "conv0_relu",
        "pool0",
        "conv1_relu",
        "pool1",
        "reshape_to_seq",
        "lstm_output",
        "lstm_last",
        "fc_output",
    ]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = []
        outputs.append(x)

        batch_size, time_steps, feature_dim = x.shape
        x = x.reshape(batch_size * time_steps, 1, feature_dim)
        outputs.append(x)

        x = torch.relu(self.model.convs[0](x))
        outputs.append(x)
        x = self.model.pools[0](x)
        outputs.append(x)

        x = torch.relu(self.model.convs[1](x))
        outputs.append(x)
        x = self.model.pools[1](x)
        outputs.append(x)

        x = x.reshape(batch_size, time_steps, -1)
        outputs.append(x)

        x, _ = self.model.lstm(x)
        outputs.append(x)

        x = x[:, -1, :]
        outputs.append(x)

        x = self.model.fc(x)
        outputs.append(x)
        return tuple(outputs)


class DebugCNNAll(nn.Module):
    """把 CNN-only 的关键中间张量全部显式返回。"""

    stage_names = [
        "input",
        "conv0_relu",
        "pool0",
        "conv1_relu",
        "pool1",
        "flatten",
        "fc_output",
    ]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = []
        outputs.append(x)

        x = torch.relu(self.model.convs[0](x))
        outputs.append(x)
        x = self.model.pools[0](x)
        outputs.append(x)

        x = torch.relu(self.model.convs[1](x))
        outputs.append(x)
        x = self.model.pools[1](x)
        outputs.append(x)

        x = x.view(x.size(0), -1)
        outputs.append(x)
        x = self.model.fc(x)
        outputs.append(x)
        return tuple(outputs)


def build_debug_wrapper(model_cfg: dict, input_shape, state_dict, root: Path):
    model_path = root / model_cfg["weights_path"]
    model_path = model_path.resolve()
    state_dict = infer.load_state_dict_compat(model_path, torch.device("cpu"))
    base_model = infer.build_model_from_config(model_cfg=model_cfg, input_shape=input_shape, state_dict=state_dict)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()
    model_type = infer.normalize_model_type(model_cfg.get("type", "CNN-All"))
    if model_type == "cnn_lstm":
        return DebugCNNLSTM(base_model), DebugCNNLSTM.stage_names
    return DebugCNNAll(base_model), DebugCNNAll.stage_names


def choose_sample(X_scaled: np.ndarray, source: np.ndarray, y: np.ndarray, sample_index: int, csv_name: Optional[str]):
    if csv_name:
        mask = source == csv_name
        if not np.any(mask):
            raise FileNotFoundError(f"鎸囧畾鐨?csv_name 鍦ㄦ牱鏈潵婧愰噷涓嶅瓨鍦? {csv_name}")
        X_scaled = X_scaled[mask]
        source = source[mask]
        y = y[mask]
    if sample_index < 0 or sample_index >= int(X_scaled.shape[0]):
        raise IndexError(f"sample_index 瓒呭嚭鑼冨洿: {sample_index}, total={X_scaled.shape[0]}")
    return X_scaled[sample_index : sample_index + 1], str(source[sample_index]), float(y[sample_index])


def export_debug_onnx(model: nn.Module, sample: np.ndarray, output_names, onnx_path: Path):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.from_numpy(sample.astype(np.float32)),
        onnx_path.as_posix(),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=None,
    )
    return onnx.load(onnx_path.as_posix())


def run_onnx_debug(model_proto, sample: np.ndarray):
    evaluator = ReferenceEvaluator(model_proto)
    input_name = model_proto.graph.input[0].name
    outputs = evaluator.run(None, {input_name: sample.astype(np.float32)})
    graph_outputs = [value.name for value in model_proto.graph.output]
    return {name: np.asarray(value) for name, value in zip(graph_outputs, outputs)}


def run_pth_debug(model: nn.Module, stage_names, sample: np.ndarray):
    with torch.no_grad():
        outputs = model(torch.from_numpy(sample.astype(np.float32)))
    return {name: value.detach().cpu().numpy() for name, value in zip(stage_names, outputs)}


def build_metrics_rows(stage_names, pth_dict, onnx_dict):
    rows = []
    for name in stage_names:
        p = np.asarray(pth_dict[name], dtype=np.float32)
        o = np.asarray(onnx_dict[name], dtype=np.float32)
        diff = o - p
        rows.append(
            [
                name,
                list(p.shape),
                float(np.mean(np.abs(diff))),
                float(np.sqrt(np.mean(diff ** 2))),
                float(np.max(np.abs(diff))),
            ]
        )
    return rows


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    infer_cfg_path = Path(args.infer_config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = root / infer_cfg_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = load_json(infer_cfg_path)
    data_cfg = infer_cfg["data"]
    data_dir = Path(args.data_dir).resolve()

    X, y, source = compare_pth_kmodel.build_dataset_with_sources(
        data_dir=data_dir,
        base_window_size=compare_pth_kmodel.require_positive_int(data_cfg["base_window_size"], "data.base_window_size"),
        base_step=compare_pth_kmodel.require_positive_int(data_cfg["base_step"], "data.base_step"),
        seq_length=compare_pth_kmodel.require_positive_int(data_cfg["sequence_length"], "data.sequence_length"),
        seq_step=compare_pth_kmodel.require_positive_int(data_cfg["sequence_step"], "data.sequence_step"),
        feature_mode=infer.normalize_feature_mode(infer_cfg.get("preprocessing", {}).get("feature_mode", "raw")),
        max_samples=None,
    )
    if X.shape[0] == 0:
        raise RuntimeError(f"娌℃湁浠庢暟鎹洰褰曟瀯寤哄嚭浠讳綍鏍锋湰: {data_dir}")

    scaler_path = (root / infer_cfg["normalization"]["scaler_path"]).resolve()
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    sample, sample_csv_name, true_label = choose_sample(
        X_scaled,
        source,
        y,
        sample_index=int(args.sample_index),
        csv_name=args.csv_name,
    )

    model_cfg = infer_cfg["model"]
    model_path = (root / model_cfg["weights_path"]).resolve()
    state_dict = infer.load_state_dict_compat(model_path, torch.device("cpu"))
    debug_model, stage_names = build_debug_wrapper(model_cfg, tuple(sample.shape[1:]), state_dict, root)
    debug_model.eval()

    pth_outputs = run_pth_debug(debug_model, stage_names, sample)

    debug_onnx_path = output_dir / "debug_model.onnx"
    model_proto = export_debug_onnx(debug_model, sample, stage_names, debug_onnx_path)
    onnx_outputs = run_onnx_debug(model_proto, sample)

    metrics_rows = build_metrics_rows(stage_names, pth_outputs, onnx_outputs)
    metrics_csv_path = output_dir / "layerwise_metrics.csv"
    save_csv(metrics_csv_path, ["stage_name", "shape", "mae", "rmse", "max_abs"], metrics_rows)

    np.savez(output_dir / "pth_tensors.npz", **pth_outputs)
    np.savez(output_dir / "onnx_tensors.npz", **onnx_outputs)

    summary = {
        "infer_config": str(infer_cfg_path),
        "data_dir": str(data_dir),
        "sample_csv_name": sample_csv_name,
        "sample_index": int(args.sample_index),
        "true_label": true_label,
        "debug_onnx_path": str(debug_onnx_path),
        "metrics_csv": str(metrics_csv_path),
        "stage_count": len(stage_names),
        "stages": [
            {
                "stage_name": row[0],
                "shape": row[1],
                "mae": row[2],
                "rmse": row[3],
                "max_abs": row[4],
            }
            for row in metrics_rows
        ],
    }
    save_json(output_dir / "layerwise_summary.json", summary)

    print("=== Layerwise PTH vs ONNX ===")
    print("sample_csv_name:", sample_csv_name)
    print("sample_index:", int(args.sample_index))
    print("true_label:", true_label)
    print("debug_onnx_path:", debug_onnx_path)
    print("metrics_csv:", metrics_csv_path)
    for row in metrics_rows:
        print(
            "stage={}, shape={}, mae={:.8f}, rmse={:.8f}, max_abs={:.8f}".format(
                row[0], row[1], row[2], row[3], row[4]
            )
        )


if __name__ == "__main__":
    main()
