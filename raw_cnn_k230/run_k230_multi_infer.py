"""
K230 单板三模型在线推理入口。

大将军，属下这里单独新增一个脚本，而不是去改老的 `run_k230_infer.py`，
核心目的只有一个：在不影响现有单模型流程的前提下，给板端增加
“一路输入，同时跑多个模型，再统一回包”的能力。

这个脚本当前聚焦在线 UART 场景，目标模型为：
1. CNN
2. CNN-TCN
3. CNN-LSTM

实现思路：
1. 只接收一路原始时序输入。
2. 维护一份原始环形缓冲区。
3. 每收到一个基础窗，就分别更新三个模型自己的输入状态。
4. 只有当三个模型都准备好时，才统一触发一次三模型推理。
5. 三个预测值按固定顺序打包后一次回传。

这样做的好处：
1. 老的单模型脚本完全不动，已有部署不受影响。
2. 新功能单独维护，定位问题更清晰。
3. 后续如果要扩成更多模型，也有清晰的扩展位。
"""

import gc

import run_k230_infer as base

try:
    import nncase_runtime as nn  # type: ignore
except ImportError:
    nn = None  # type: ignore


DEFAULT_MULTI_CONFIG_PATH = "configs/k230_config_multi.json"
OVERRIDE_CONFIG_PATH = None

MULTI_RUNTIME_CACHE = {
    "dataset_key": None,
    "X_shared": None,
    "y": None,
    "cursor": 0,
}


def require_field(obj, field_name, context_name):
    """中文注释：读取必填字段，缺失时直接报清晰错误，避免板端默默跑偏。"""
    if field_name not in obj:
        raise ValueError("{} missing required field '{}'".format(context_name, field_name))
    return obj[field_name]


def resolve_multi_config_path(root, cli_args):
    """
    中文注释：多模型脚本沿用老脚本的命令行约定。

    支持：
    1. `python run_k230_multi_infer.py`
    2. `python run_k230_multi_infer.py --config configs/xxx.json`
    3. `python run_k230_multi_infer.py configs/xxx.json`
    """
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
        if token.lower().endswith(".json") or token.lower().endswith(".jsonc"):
            selected = token
            break
        idx += 1

    if OVERRIDE_CONFIG_PATH:
        selected = str(OVERRIDE_CONFIG_PATH)

    if not selected:
        return base.join_path(root, DEFAULT_MULTI_CONFIG_PATH)
    if base.is_abs_path(selected):
        return base.norm_path(selected)
    return base.join_path(root, selected)


class ModelRuntimeContext:
    """
    中文注释：单个模型的运行时上下文。

    每个模型都维护自己的一套状态，避免三模型之间互相覆盖：
    1. 自己的 kmodel / scaler / KPU 实例
    2. 自己的输入时序缓存
    3. 自己最近一次预测结果
    4. 自己是否已经准备好参与统一推理
    """

    def __init__(self, root, model_cfg):
        self.root = root
        self.name = str(require_field(model_cfg, "name", "model")).strip()
        self.model_type = base.normalize_model_type(require_field(model_cfg, "type", "model"))
        if self.model_type not in {"cnn", "cnn_tcn", "cnn_lstm"}:
            raise ValueError("Unsupported multi-model type: {}".format(self.model_type))

        data_cfg = require_field(model_cfg, "data", "model")
        paths_cfg = require_field(model_cfg, "paths", "model")
        preprocessing_cfg = model_cfg.get("preprocessing", {})

        self.window_size = base.require_positive_int(
            require_field(data_cfg, "base_window_size", "model.data"),
            "{}.data.base_window_size".format(self.name),
        )
        self.base_step = base.resolve_positive_step(
            data_cfg.get("base_step", None),
            self.window_size // 2,
            "{}.data.base_step".format(self.name),
        )
        self.sequence_length = base.require_positive_int(
            require_field(data_cfg, "sequence_length", "model.data"),
            "{}.data.sequence_length".format(self.name),
        )
        self.sequence_step = base.require_positive_int(
            data_cfg.get("sequence_step", 1),
            "{}.data.sequence_step".format(self.name),
        )
        self.feature_mode = base.normalize_feature_mode(preprocessing_cfg.get("feature_mode", "raw"))

        self.kmodel_path = base.join_path(root, require_field(paths_cfg, "kmodel", "model.paths"))
        self.scaler_json_path = base.join_path(root, require_field(paths_cfg, "scaler_json", "model.paths"))

        self.mean, self.scale = base.load_scaler_params(self.scaler_json_path)
        if len(self.mean) != self.window_size or len(self.scale) != self.window_size:
            raise RuntimeError(
                "scaler length mismatch for {}: need {}, got mean={}, scale={}".format(
                    self.name,
                    self.window_size,
                    len(self.mean),
                    len(self.scale),
                )
            )

        if nn is None:
            raise RuntimeError("nncase_runtime is not available, cannot start multi-model inference.")
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(self.kmodel_path)

        # 中文注释：CNN 只吃当前窗，因此不需要序列环形缓存。
        self.seq_ring = None
        self.seq_write_idx = 0
        self.seq_filled = 0
        self.seq_windows_since_infer = 0
        if self.sequence_length > 1:
            self.seq_ring = base.empty_float((self.sequence_length, self.window_size))

        self.last_pred = 0.0
        self.infer_count = 0
        self.total_infer_us = 0
        self.ready = False
        self.just_became_ready = False

    def update_with_base_window(self, proc_window, tmp_seq):
        """
        中文注释：收到一个新的基础窗后，更新当前模型自己的输入状态。

        这里不直接推理，只负责回答一个问题：
        “这个模型在这一轮，是否已经准备好参与统一推理？”
        """
        scaled_window = (proc_window - self.mean) / self.scale
        self.just_became_ready = False

        if self.sequence_length <= 1:
            self.ready = True
            self.just_became_ready = True
            return

        self.seq_ring[self.seq_write_idx] = scaled_window
        self.seq_write_idx += 1
        if self.seq_write_idx >= self.sequence_length:
            self.seq_write_idx = 0

        if self.seq_filled < self.sequence_length:
            self.seq_filled += 1
            if self.seq_filled >= self.sequence_length:
                self.ready = True
                self.just_became_ready = True
                self.seq_windows_since_infer = 0
            else:
                self.ready = False
            return

        self.seq_windows_since_infer += 1
        if self.seq_windows_since_infer >= self.sequence_step:
            self.ready = True
            self.seq_windows_since_infer = 0
        else:
            self.ready = False

        if self.ready and self.seq_ring is not None:
            base.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)

    def build_input_sample(self, proc_window, tmp_seq, sample3d):
        """
        中文注释：把当前模型的输入组织成 KPU 需要的 `(1, T, W)` 形状。

        - CNN:       T=1
        - CNN-TCN:   T=5
        - CNN-LSTM:  T=5
        """
        if self.sequence_length <= 1:
            sample3d.reshape((1, 1, self.window_size))[0, 0, :] = (proc_window - self.mean) / self.scale
            return sample3d.reshape((1, 1, self.window_size))

        base.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)
        sample3d.reshape((1, self.sequence_length, self.window_size))[0, :, :] = tmp_seq
        return sample3d.reshape((1, self.sequence_length, self.window_size))

    def run_inference(self, proc_window, tmp_seq, sample3d):
        """中文注释：执行一次单模型推理，并记录耗时与最近结果。"""
        sample = self.build_input_sample(proc_window, tmp_seq, sample3d)
        input_tensor = nn.from_numpy(base.astype_float_array(sample))
        self.kpu.set_input_tensor(0, input_tensor)
        t0 = base.now_us()
        self.kpu.run()
        t1 = base.now_us()
        output = self.kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        self.last_pred = pred
        self.infer_count += 1
        self.total_infer_us += base.diff_us(t1, t0)
        del output
        del input_tensor
        return pred, base.diff_us(t1, t0)


class ModelInputRuntimeContext:
    """
    中文注释：单个“模型 + 输入路”的在线状态。

    模型本体和 KPU 实例由 ModelRuntimeContext 持有；这里单独维护某一路输入
    对应该模型的序列缓存，避免两路数据互相覆盖，同时也避免同一 kmodel 重复加载。
    """

    def __init__(self, model_ctx, input_ctx, output_name):
        self.model_ctx = model_ctx
        self.input_ctx = input_ctx
        self.output_name = str(output_name)
        self.seq_ring = None
        self.seq_write_idx = 0
        self.seq_filled = 0
        self.seq_windows_since_infer = 0
        if self.model_ctx.sequence_length > 1:
            self.seq_ring = base.empty_float((self.model_ctx.sequence_length, self.model_ctx.window_size))
        self.ready = False
        self.just_became_ready = False
        self.last_pred = 0.0
        self.infer_count = 0
        self.total_infer_us = 0

    def update_with_base_window(self, proc_window):
        # 中文注释：每一路输入都独立维护自己的序列状态。
        scaled_window = (proc_window - self.model_ctx.mean) / self.model_ctx.scale
        self.just_became_ready = False

        if self.model_ctx.sequence_length <= 1:
            self.ready = True
            self.just_became_ready = True
            return

        self.seq_ring[self.seq_write_idx] = scaled_window
        self.seq_write_idx += 1
        if self.seq_write_idx >= self.model_ctx.sequence_length:
            self.seq_write_idx = 0

        if self.seq_filled < self.model_ctx.sequence_length:
            self.seq_filled += 1
            if self.seq_filled >= self.model_ctx.sequence_length:
                self.ready = True
                self.just_became_ready = True
                self.seq_windows_since_infer = 0
            else:
                self.ready = False
            return

        self.seq_windows_since_infer += 1
        if self.seq_windows_since_infer >= self.model_ctx.sequence_step:
            self.ready = True
            self.seq_windows_since_infer = 0
        else:
            self.ready = False

    def build_input_sample(self, proc_window, tmp_seq, sample3d):
        if self.model_ctx.sequence_length <= 1:
            sample3d.reshape((1, 1, self.model_ctx.window_size))[0, 0, :] = (
                proc_window - self.model_ctx.mean
            ) / self.model_ctx.scale
            return sample3d.reshape((1, 1, self.model_ctx.window_size))

        base.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)
        sample3d.reshape((1, self.model_ctx.sequence_length, self.model_ctx.window_size))[0, :, :] = tmp_seq
        return sample3d.reshape((1, self.model_ctx.sequence_length, self.model_ctx.window_size))

    def run_inference(self, proc_window, tmp_seq, sample3d):
        sample = self.build_input_sample(proc_window, tmp_seq, sample3d)
        input_tensor = nn.from_numpy(base.astype_float_array(sample))
        self.model_ctx.kpu.set_input_tensor(0, input_tensor)
        t0 = base.now_us()
        self.model_ctx.kpu.run()
        t1 = base.now_us()
        output = self.model_ctx.kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        infer_us = base.diff_us(t1, t0)
        self.last_pred = pred
        self.infer_count += 1
        self.total_infer_us += infer_us
        self.model_ctx.last_pred = pred
        self.model_ctx.infer_count += 1
        self.model_ctx.total_infer_us += infer_us
        del output
        del input_tensor
        return pred, infer_us


def parse_multi_inputs(cfg, channel_count):
    """
    中文注释：解析有效输入路配置。

    未配置 inputs 时默认只使用第 0 路，保持旧版“一路多模型”的行为。
    """
    raw_inputs = cfg.get("inputs", None)
    if raw_inputs is None:
        raw_inputs = [{"name": "ch0", "source_index": 0}]
    if not isinstance(raw_inputs, list) or len(raw_inputs) <= 0:
        raise ValueError("config.inputs must be a non-empty list.")

    inputs = []
    used_names = set()
    for idx, item in enumerate(raw_inputs):
        if isinstance(item, dict):
            name = str(item.get("name", "ch{}".format(idx))).strip()
            source_index = int(item.get("source_index", idx))
        else:
            name = str(item).strip()
            source_index = idx
        if not name:
            name = "ch{}".format(idx)
        if name in used_names:
            raise ValueError("Duplicate input name in config.inputs: {}".format(name))
        if source_index < 0 or source_index >= int(channel_count):
            raise ValueError(
                "Input {} source_index={} out of range 0..{}".format(
                    name,
                    source_index,
                    int(channel_count) - 1,
                )
            )
        used_names.add(name)
        inputs.append({"name": name, "source_index": source_index})
    return inputs


def normalize_name_list(raw_value):
    # 中文注释：兼容字符串和字符串列表两种配置写法。
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if str(v).strip()]
    text = str(raw_value).strip()
    if not text:
        return []
    return [text]


def make_model_input_bindings(model_contexts, model_cfgs, input_contexts):
    """
    中文注释：把 models 和 inputs 展开成“模型 + 输入路”的运行项。

    每个 model 可配置：
    - input: "ch0"
    - inputs: ["ch0", "ch1"]

    如果未配置 input/inputs，则默认绑定所有有效输入路；只有一路时保持旧输出名。
    """
    input_by_name = {}
    for item in input_contexts:
        input_by_name[item["name"]] = item

    multi_input = len(input_contexts) > 1
    bindings = []
    used_output_names = set()
    for model_ctx, model_cfg in zip(model_contexts, model_cfgs):
        names = normalize_name_list(model_cfg.get("inputs", None))
        if not names:
            names = normalize_name_list(model_cfg.get("input", None))
        if not names:
            names = [item["name"] for item in input_contexts]

        for input_name in names:
            if input_name not in input_by_name:
                raise ValueError("Model {} references unknown input '{}'.".format(model_ctx.name, input_name))
            output_name = model_ctx.name
            if multi_input or len(names) > 1:
                output_name = "{}_{}".format(model_ctx.name, input_name)
            if "output_name" in model_cfg and len(names) == 1:
                output_name = str(model_cfg.get("output_name"))
            if output_name in used_output_names:
                raise ValueError("Duplicate model output name: {}".format(output_name))
            used_output_names.add(output_name)
            bindings.append(ModelInputRuntimeContext(model_ctx, input_by_name[input_name], output_name))

    if not bindings:
        raise ValueError("No model/input bindings configured for multi-model runtime.")
    return bindings


def adapt_shared_sample_for_model(model_ctx, shared_sample):
    # 中文注释：多模型离线模式统一按“最大 sequence_length”构造样本，
    # 这里再把共享样本裁剪成当前模型真正需要的片段。
    if model_ctx.sequence_length <= 1:
        return shared_sample[-1:, :]
    return shared_sample[-model_ctx.sequence_length :, :]


def scale_sample_for_model(model_ctx, model_sample, out_sample):
    """中文注释：csv_cached 共享样本是未标准化特征，推理前必须按当前模型自己的 scaler 标准化。"""
    out_sample[:] = (model_sample - model_ctx.mean) / model_ctx.scale
    return out_sample


def run_prebuilt_sample(model_ctx, sample):
    # 中文注释：给 csv_cached 用的通用推理入口。
    # 外部先准备好当前模型应该吃的样本，这里只负责跑 KPU 并记录统计。
    sample3d = base.astype_float_array(sample).reshape((1, sample.shape[0], sample.shape[1]))
    input_tensor = nn.from_numpy(sample3d)
    model_ctx.kpu.set_input_tensor(0, input_tensor)
    t0 = base.now_us()
    model_ctx.kpu.run()
    t1 = base.now_us()
    output = model_ctx.kpu.get_output_tensor(0)
    pred = float(output.to_numpy().reshape(-1)[0])
    model_ctx.last_pred = pred
    model_ctx.infer_count += 1
    model_ctx.total_infer_us += base.diff_us(t1, t0)
    del output
    del input_tensor
    return pred, base.diff_us(t1, t0)


def validate_multi_models(model_contexts):
    """
    中文注释：多模型版为了尽量稳定，先限制几个关键前提必须一致。

    这样可以把多模型调度逻辑压缩成一条主数据流，避免一上来就把复杂度拉满。
    当前要求统一的参数：
    1. base_window_size
    2. base_step
    3. feature_mode
    """
    if not model_contexts:
        raise ValueError("No models configured for multi-model runtime.")

    first = model_contexts[0]
    for ctx in model_contexts[1:]:
        if ctx.window_size != first.window_size:
            raise ValueError("All models must use the same base_window_size in multi-model mode.")
        if ctx.base_step != first.base_step:
            raise ValueError("All models must use the same base_step in multi-model mode.")
        if ctx.feature_mode != first.feature_mode:
            raise ValueError("All models must use the same preprocessing.feature_mode in multi-model mode.")


def send_zero_frame(uart_sender):
    """中文注释：未就绪时按新协议发一帧全零，方便上位机保持节拍。"""
    uart_sender.send_values_frame([0.0] * int(uart_sender.value_count))


def build_multi_output_frame(model_pred_map, uart_sender, output_cfg):
    """中文注释：按配置里的输出槽位组包，未配置或未命中的槽位补默认值。"""
    value_count = int(uart_sender.value_count)
    fill_value = float(output_cfg.get("fill_value", 0.0))
    slots = output_cfg.get("slots", [])

    values = [fill_value] * value_count
    for i in range(value_count):
        if i >= len(slots):
            break
        model_name = slots[i]
        if model_name is None:
            continue
        model_name = str(model_name)
        if model_name in model_pred_map:
            values[i] = float(model_pred_map[model_name])
    return values


def get_common_runtime_shape(model_contexts):
    # 中文注释：返回多模型公共基础窗参数，以及统一样本要用的最大序列长度。
    first = model_contexts[0]
    max_seq_length = 1
    for ctx in model_contexts:
        if ctx.sequence_length > max_seq_length:
            max_seq_length = ctx.sequence_length
    return first.window_size, first.base_step, first.feature_mode, max_seq_length


def make_shared_dataset_cache_key(root, test_data_dir, model_contexts, max_samples):
    # 中文注释：给多模型 csv 样本缓存生成键，关键输入变了就重建缓存。
    window_size, base_step, feature_mode, max_seq_length = get_common_runtime_shape(model_contexts)
    files = base.list_csv_files(test_data_dir)
    parts = [
        base.norm_path(root),
        base.norm_path(test_data_dir),
        str(window_size),
        str(base_step),
        str(feature_mode),
        str(max_seq_length),
        str(max_samples),
    ]
    for path in files:
        size, mtime = base.file_size_mtime(path)
        parts.append("{}:{}:{}".format(base.norm_path(path), size, mtime))
    return "|".join(parts)


def build_shared_dataset(cfg, root, model_contexts, max_samples):
    # 中文注释：统一按最大 sequence_length 构造共享样本，
    # 这样 csv 模式下三模型共享同一套样本时刻定义。
    paths_cfg = require_field(cfg, "paths", "config")
    test_data_dir = require_field(paths_cfg, "test_data_dir", "config.paths")
    window_size, base_step, feature_mode, max_seq_length = get_common_runtime_shape(model_contexts)
    shared_cfg = {
        "paths": {
            "test_data_dir": test_data_dir,
        },
        "data": {
            "base_window_size": window_size,
            "base_step": base_step,
            "sequence_length": max_seq_length,
            "sequence_step": 1,
        },
        "preprocessing": {
            "feature_mode": feature_mode,
        },
    }
    return base.build_dataset(shared_cfg, root=root, max_samples=max_samples)


def ensure_shared_dataset_cache(cfg, root, model_contexts, max_samples):
    # 中文注释：构造或命中多模型 csv 共享样本缓存。
    paths_cfg = require_field(cfg, "paths", "config")
    test_data_dir = base.join_path(root, require_field(paths_cfg, "test_data_dir", "config.paths"))
    cache_key = make_shared_dataset_cache_key(root, test_data_dir, model_contexts, max_samples)
    if (
        MULTI_RUNTIME_CACHE.get("dataset_key", None) == cache_key
        and MULTI_RUNTIME_CACHE.get("X_shared", None) is not None
        and MULTI_RUNTIME_CACHE.get("y", None) is not None
    ):
        return MULTI_RUNTIME_CACHE["X_shared"], MULTI_RUNTIME_CACHE["y"], False

    X_shared, y = build_shared_dataset(cfg, root, model_contexts, max_samples)
    MULTI_RUNTIME_CACHE["dataset_key"] = cache_key
    MULTI_RUNTIME_CACHE["X_shared"] = X_shared
    MULTI_RUNTIME_CACHE["y"] = y
    MULTI_RUNTIME_CACHE["cursor"] = 0
    return X_shared, y, True


def acquire_multi_infer_range(total_samples, request_count):
    # 中文注释：多模型 csv 模式下按游标抓取下一批样本。
    if total_samples <= 0:
        raise RuntimeError("No cached shared samples available.")
    count = int(request_count)
    if count <= 0:
        count = 1
    if count > total_samples:
        count = total_samples

    start_idx = int(MULTI_RUNTIME_CACHE.get("cursor", 0)) % int(total_samples)
    next_cursor = start_idx + count
    while next_cursor >= total_samples:
        next_cursor -= total_samples
    MULTI_RUNTIME_CACHE["cursor"] = next_cursor
    return start_idx, count


def collect_labels_range(y_all, start_idx, count):
    # 中文注释：取出当前批次对应标签。
    total = int(len(y_all))
    out = base.empty_float((count,))
    idx = int(start_idx)
    for i in range(count):
        out[i] = y_all[idx]
        idx += 1
        if idx >= total:
            idx = 0
    return out


def collect_shared_sample_range(X_shared, start_idx, count):
    # 中文注释：取出当前批次共享样本，避免后续推理时再处理游标回绕。
    total = int(X_shared.shape[0])
    seq_len = int(X_shared.shape[1])
    width = int(X_shared.shape[2])
    out = base.empty_float((count, seq_len, width))
    idx = int(start_idx)
    for i in range(count):
        out[i] = X_shared[idx]
        idx += 1
        if idx >= total:
            idx = 0
    return out


def write_multi_predictions(path, y_true, pred_map, model_names):
    # 中文注释：输出多模型合并预测结果，便于直接横向对比三模型。
    out_path = base.norm_path(path)
    out_dir = base.dirname(out_path)
    if out_dir not in {"", "."}:
        base.ensure_dir(out_dir)
    with open(out_path, "w") as f:
        header = ["sample_id", "true_label"]
        for model_name in model_names:
            header.append("pred_{}".format(model_name))
        f.write(",".join(header) + "\n")
        for i in range(len(y_true)):
            row = [str(i), str(float(y_true[i]))]
            for model_name in model_names:
                row.append(str(float(pred_map[model_name][i])))
            f.write(",".join(row) + "\n")


def run_multi_uart_online(cfg, root, uart_sender):
    """
    中文注释：单板多模型在线推理主循环。

    数据流分三层：
    1. 原始点流 -> 原始环形缓冲
    2. 原始环形缓冲 -> 基础窗
    3. 基础窗 -> 三个模型各自的输入缓存 -> 三模型统一推理
    """
    runtime_cfg = cfg.get("runtime", {})
    online_cfg = base.get_runtime_section(runtime_cfg, "uart_online")
    output_cfg = base.get_runtime_section(runtime_cfg, "output")
    alarm_cfg = base.get_runtime_section(runtime_cfg, "full_gas_alarm")
    model_cfgs = cfg.get("models", [])
    model_contexts = [ModelRuntimeContext(root, item) for item in model_cfgs]
    validate_multi_models(model_contexts)

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; multi-model uart_online mode cannot start.")

    common = model_contexts[0]
    window_size = common.window_size
    base_step = common.base_step
    feature_mode = common.feature_mode
    max_seq_length = get_common_runtime_shape(model_contexts)[3]
    zero_guard_cfg = base.get_zero_guard_config(cfg)
    zero_guard_enabled = bool(zero_guard_cfg.get("enabled", False))
    zero_guard_output_value = float(zero_guard_cfg.get("output_value", 0.0))
    full_gas_alarm = base.FullGasAlarmState(alarm_cfg)

    channel_count = base.require_positive_int(
        online_cfg.get("channel_count", 1),
        "runtime.uart_online.channel_count",
    )
    input_contexts = parse_multi_inputs(cfg, channel_count)
    model_bindings = make_model_input_bindings(model_contexts, model_cfgs, input_contexts)
    postprocessor = base.create_runtime_postprocessor(cfg, channel_count=len(model_bindings))
    input_index_by_name = {}
    for idx, item in enumerate(input_contexts):
        input_index_by_name[item["name"]] = idx

    idle_sleep_ms = int(online_cfg.get("idle_sleep_ms", 1))
    warmup_send = bool(online_cfg.get("send_zeros_before_ready", False))
    quiet = bool(online_cfg.get("quiet", False))
    debug_predict_trace = bool(online_cfg.get("debug_predict_trace", True))
    debug_uart_read_timing = bool(online_cfg.get("debug_uart_read_timing", False))
    debug_outer_rx = bool(online_cfg.get("debug_outer_rx", False))
    debug_outer_rx_only_abnormal = bool(online_cfg.get("debug_outer_rx_only_abnormal", False))
    debug_outer_rx_interval_warn_ms = float(online_cfg.get("debug_outer_rx_interval_warn_ms", 25.0))
    debug_tx_timing = bool(online_cfg.get("debug_tx_timing", False))
    debug_tx_only_abnormal = bool(online_cfg.get("debug_tx_only_abnormal", True))
    debug_tx_interval_min_warn_ms = float(online_cfg.get("debug_tx_interval_min_warn_ms", 180.0))
    debug_tx_interval_max_warn_ms = float(online_cfg.get("debug_tx_interval_max_warn_ms", 240.0))
    flush_rx_on_start = bool(online_cfg.get("flush_rx_on_start", True))
    startup_flush_empty_rounds = int(online_cfg.get("startup_flush_empty_rounds", 3))
    startup_flush_sleep_ms = int(online_cfg.get("startup_flush_sleep_ms", 10))

    input_value_type = str(online_cfg.get("input_value_type", "int32")).lower()
    input_byte_order = str(online_cfg.get("input_byte_order", uart_sender.byte_order)).lower()
    if uart_sender.outer_frame_enabled:
        parser = base.UartBundledValueFrameParser(
            outer_header=uart_sender.outer_header,
            outer_tail=uart_sender.outer_tail,
            inner_header=uart_sender.header,
            inner_tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
            outer_frame_count=uart_sender.outer_frame_count,
        )
    else:
        parser = base.UartValueFrameParser(
            header=uart_sender.header,
            tail=uart_sender.tail,
            value_count=channel_count,
            value_type=input_value_type,
            byte_order=input_byte_order,
        )

    # 中文注释：每个有效输入路各维护一份原始点缓存，模型本体仍然共享，避免重复加载 kmodel。
    raw_ring = base.empty_float((len(input_contexts), window_size))
    raw_write_idx = 0
    raw_filled_frames = 0
    raw_frames_since_emit = 0

    total_rx_frames = 0
    total_tx_frames = 0
    base_window_count = 0
    infer_round = 0

    # 中文注释：这些临时数组反复复用，避免板端在 while True 里频繁申请内存。
    tmp_window = base.empty_float((window_size,))
    tmp_feature_map = {}
    for input_ctx in input_contexts:
        tmp_feature_map[input_ctx["name"]] = base.empty_float((window_size,))
    tmp_seq_map = {}
    tmp_sample_map = {}
    for binding in model_bindings:
        if binding.model_ctx.sequence_length > 1:
            tmp_seq_map[binding.output_name] = base.empty_float((binding.model_ctx.sequence_length, window_size))
            tmp_sample_map[binding.output_name] = base.empty_float((1, binding.model_ctx.sequence_length, window_size))
        else:
            tmp_seq_map[binding.output_name] = None
            tmp_sample_map[binding.output_name] = base.empty_float((1, 1, window_size))
    zero_seq_ring = base.empty_float((len(input_contexts), max_seq_length, window_size)) if zero_guard_enabled else None
    tmp_zero_seq = base.empty_float((max_seq_length, window_size)) if zero_guard_enabled else None
    zero_seq_write_idx = 0
    zero_seq_filled = 0

    def online_print(*args):
        if not quiet:
            print(*args)

    online_print("=== K230 Multi-Model Runtime ===")
    online_print("config_name:", cfg.get("name", ""))
    online_print("mode:", "uart_online")
    online_print("model_count:", len(model_contexts))
    online_print("input_count:", len(input_contexts))
    online_print(
        "models:",
        ", ".join("{}({})".format(ctx.name, ctx.model_type) for ctx in model_contexts),
    )
    online_print(
        "inputs:",
        ", ".join("{}<=values[{}]".format(item["name"], item["source_index"]) for item in input_contexts),
    )
    online_print(
        "outputs:",
        ", ".join(binding.output_name for binding in model_bindings),
    )
    online_print(
        "postprocessing: enabled={}, type={}".format(
            bool(postprocessor.enabled),
            postprocessor.kind,
        )
    )
    online_print(
        "uart_online_cfg: channels={}, window={}, base_step={}, feature_mode={}, output_value_count={}".format(
            channel_count,
            window_size,
            base_step,
            feature_mode,
            uart_sender.value_count,
        )
    )
    online_print(
        "uart_online_zero_guard: enabled={}, output_value={}, min_votes={}".format(
            bool(zero_guard_enabled),
            zero_guard_output_value,
            int(zero_guard_cfg.get("min_votes", 3)),
        )
    )
    online_print("uart_online_full_gas_alarm:", full_gas_alarm.summary())

    if flush_rx_on_start:
        flushed_bytes = base.drain_uart_rx(
            uart_sender.uart,
            empty_rounds=startup_flush_empty_rounds,
            sleep_between_ms=startup_flush_sleep_ms,
        )
        online_print(
            "uart_online_startup_flush: enabled=True, flushed_bytes={}, empty_rounds={}, sleep_ms={}".format(
                flushed_bytes,
                startup_flush_empty_rounds,
                startup_flush_sleep_ms,
            )
        )
    else:
        online_print("uart_online_startup_flush: enabled=False")

    session_start_us = base.now_us()
    first_rx_us = None
    last_infer_trigger_us = None
    last_uart_read_us = None
    last_outer_rx_us = None
    last_small_rx_us = None
    last_tx_us = None

    while True:
        raw = uart_sender.uart.read()
        rx_now_us = base.now_us()
        if debug_uart_read_timing and not quiet:
            read_interval_ms = -1.0
            if last_uart_read_us is not None:
                read_interval_ms = base.diff_us(rx_now_us, last_uart_read_us) / 1000.0
            raw_len = len(raw) if raw else 0
            online_print(
                "uart_online_read: ts_ms={:.3f}, interval_ms={:.3f}, raw_bytes={}, has_data={}".format(
                    rx_now_us / 1000.0,
                    read_interval_ms,
                    raw_len,
                    bool(raw),
                )
            )
        last_uart_read_us = rx_now_us

        if not raw:
            base.sleep_ms(idle_sleep_ms)
            continue

        frames = parser.feed(raw)
        if not frames:
            continue

        if uart_sender.outer_frame_enabled:
            outer_count = int(uart_sender.outer_frame_count)
            parsed_outer_frames = len(frames) // outer_count
            if parsed_outer_frames > 0:
                outer_interval_ms = -1.0
                if last_outer_rx_us is not None:
                    outer_interval_ms = base.diff_us(rx_now_us, last_outer_rx_us) / 1000.0
                if debug_outer_rx:
                    need_print_outer_rx = True
                    if debug_outer_rx_only_abnormal:
                        need_print_outer_rx = parsed_outer_frames > 1
                        if not need_print_outer_rx and outer_interval_ms >= 0.0:
                            need_print_outer_rx = outer_interval_ms >= debug_outer_rx_interval_warn_ms
                    if need_print_outer_rx:
                        online_print(
                            "uart_online_outer_rx: ts_ms={:.3f}, outer_frame_idx={}, batch_outer_frames={}, interval_ms={:.3f}, raw_bytes={}, parsed_small_frames={}".format(
                                rx_now_us / 1000.0,
                                (total_rx_frames + len(frames)) // outer_count,
                                parsed_outer_frames,
                                outer_interval_ms,
                                len(raw),
                                len(frames),
                            )
                        )
                last_outer_rx_us = rx_now_us
        else:
            small_interval_ms = -1.0
            if last_small_rx_us is not None:
                small_interval_ms = base.diff_us(rx_now_us, last_small_rx_us) / 1000.0
            if debug_outer_rx:
                need_print_small_rx = True
                if debug_outer_rx_only_abnormal:
                    need_print_small_rx = len(frames) > 1
                    if not need_print_small_rx and small_interval_ms >= 0.0:
                        need_print_small_rx = small_interval_ms >= debug_outer_rx_interval_warn_ms
                if need_print_small_rx:
                    online_print(
                        "uart_online_small_rx: ts_ms={:.3f}, small_frame_idx={}, batch_small_frames={}, interval_ms={:.3f}, raw_bytes={}".format(
                            rx_now_us / 1000.0,
                            total_rx_frames + len(frames),
                            len(frames),
                            small_interval_ms,
                            len(raw),
                        )
                    )
            last_small_rx_us = rx_now_us

        for values in frames:
            # 中文注释：按 inputs 配置抽取有效输入路；每一路后续都独立判定，不做融合。
            total_rx_frames += 1
            if first_rx_us is None:
                first_rx_us = base.now_us()

            for input_idx, input_ctx in enumerate(input_contexts):
                raw_ring[input_idx][raw_write_idx] = float(values[int(input_ctx["source_index"])])
            raw_write_idx += 1
            if raw_write_idx >= window_size:
                raw_write_idx = 0

            emit_base_window = False
            first_base_window = False
            if raw_filled_frames < window_size:
                raw_filled_frames += 1
                if raw_filled_frames >= window_size:
                    emit_base_window = True
                    first_base_window = True
                    raw_frames_since_emit = 0
                else:
                    if warmup_send:
                        send_zero_frame(uart_sender)
                        total_tx_frames += 1
                    continue
            else:
                raw_frames_since_emit += 1
                if raw_frames_since_emit >= base_step:
                    emit_base_window = True
                    raw_frames_since_emit = 0
                else:
                    if warmup_send:
                        send_zero_frame(uart_sender)
                        total_tx_frames += 1
                    continue

            if not emit_base_window:
                continue

            freq_total = 0.0
            for input_idx, input_ctx in enumerate(input_contexts):
                base.expand_ring_window(raw_ring[input_idx], raw_write_idx, tmp_window)
                freq_total += base.mean_1d(tmp_window)
                if zero_guard_enabled:
                    zero_seq_ring[input_idx][zero_seq_write_idx] = tmp_window
                base.apply_feature_mode_1d(tmp_window, feature_mode, tmp_feature_map[input_ctx["name"]])
            freq_mean = freq_total / float(len(input_contexts))
            if zero_guard_enabled:
                zero_seq_write_idx += 1
                if zero_seq_write_idx >= max_seq_length:
                    zero_seq_write_idx = 0
                if zero_seq_filled < max_seq_length:
                    zero_seq_filled += 1
            base_window_count += 1

            ready_count = 0
            for binding in model_bindings:
                binding.update_with_base_window(tmp_feature_map[binding.input_ctx["name"]])
                if binding.ready:
                    ready_count += 1

            if ready_count != len(model_bindings):
                if warmup_send:
                    send_zero_frame(uart_sender)
                    total_tx_frames += 1
                continue

            if debug_predict_trace:
                trigger_now_us = base.now_us()
                elapsed_from_start_ms = base.diff_us(trigger_now_us, session_start_us) / 1000.0
                elapsed_from_first_rx_ms = -1.0
                if first_rx_us is not None:
                    elapsed_from_first_rx_ms = base.diff_us(trigger_now_us, first_rx_us) / 1000.0
                since_last_infer_ms = -1.0
                if last_infer_trigger_us is not None:
                    since_last_infer_ms = base.diff_us(trigger_now_us, last_infer_trigger_us) / 1000.0
                last_infer_trigger_us = trigger_now_us
                online_print(
                    "uart_online_trigger: infer_round_next={}, rx_small_frame_idx={}, base_window_idx={}, first_base_window={}, ready_models={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
                        infer_round + 1,
                        total_rx_frames,
                        base_window_count,
                        first_base_window,
                        len(model_bindings),
                        elapsed_from_start_ms,
                        elapsed_from_first_rx_ms,
                        since_last_infer_ms,
                    )
                )

            infer_round += 1
            model_values = []
            model_pred_map = {}
            infer_costs_ms = []
            zero_guard_hit = False
            zero_guard_votes = 0
            zero_guard_features = {}
            for binding_idx, binding in enumerate(model_bindings):
                binding_zero_guard_hit = False
                if zero_guard_enabled and zero_seq_filled >= max_seq_length:
                    input_idx = input_index_by_name[binding.input_ctx["name"]]
                    base.expand_sequence_ring(zero_seq_ring[input_idx], zero_seq_write_idx, tmp_zero_seq)
                    guard_scaled_seq = None
                    if binding.model_ctx.sequence_length > 1:
                        base.expand_sequence_ring(
                            binding.seq_ring,
                            binding.seq_write_idx,
                            tmp_seq_map[binding.output_name],
                        )
                        guard_scaled_seq = tmp_seq_map[binding.output_name]
                    binding_zero_guard_hit, zero_guard_votes, zero_guard_features = base.is_zero_guard_hit(
                        tmp_zero_seq,
                        guard_scaled_seq,
                        zero_guard_cfg,
                    )
                if binding_zero_guard_hit:
                    pred = postprocessor.update(
                        binding_idx,
                        zero_guard_output_value,
                        zero_guard_hit=True,
                    )
                    binding.last_pred = pred
                    binding.model_ctx.last_pred = pred
                    infer_us = 0
                    zero_guard_hit = True
                    if debug_predict_trace:
                        online_print(
                            "uart_online_zero_guard_hit: infer_round={}, output={}, votes={}, features={}".format(
                                infer_round,
                                binding.output_name,
                                int(zero_guard_votes),
                                {
                                    "diff_p95_abs": round(float(zero_guard_features.get("diff_p95_abs", 0.0)), 3),
                                    "win_range_mean": round(float(zero_guard_features.get("win_range_mean", 0.0)), 3),
                                    "win_std_mean": round(float(zero_guard_features.get("win_std_mean", 0.0)), 3),
                                    "absz_mean": round(float(zero_guard_features.get("absz_mean", 0.0)), 6),
                                },
                            )
                        )
                else:
                    pred, infer_us = binding.run_inference(
                        tmp_feature_map[binding.input_ctx["name"]],
                        tmp_seq_map[binding.output_name],
                        tmp_sample_map[binding.output_name],
                    )
                    pred = postprocessor.update(binding_idx, pred)
                    binding.last_pred = pred
                    binding.model_ctx.last_pred = pred
                model_values.append(pred)
                model_pred_map[binding.output_name] = pred
                infer_costs_ms.append(infer_us / 1000.0)

            if full_gas_alarm.enabled:
                alarm_value = full_gas_alarm.update(
                    model_values,
                    freq_mean,
                    zero_guard_hit=zero_guard_hit,
                )
                model_pred_map[full_gas_alarm.output_name] = alarm_value

            tx_values = build_multi_output_frame(model_pred_map, uart_sender, output_cfg)
            uart_sender.send_values_frame(tx_values)
            total_tx_frames += 1

            if debug_tx_timing:
                tx_now_us = base.now_us()
                tx_interval_ms = -1.0
                if last_tx_us is not None:
                    tx_interval_ms = base.diff_us(tx_now_us, last_tx_us) / 1000.0
                need_print_tx = True
                if debug_tx_only_abnormal:
                    need_print_tx = False
                    if tx_interval_ms >= 0.0:
                        need_print_tx = (
                            tx_interval_ms <= debug_tx_interval_min_warn_ms
                            or tx_interval_ms >= debug_tx_interval_max_warn_ms
                        )
                if need_print_tx:
                    online_print(
                        "uart_online_tx: ts_ms={:.3f}, tx_small_frame_idx={}, infer_round={}, interval_since_last_tx_ms={:.3f}, values={}".format(
                            tx_now_us / 1000.0,
                            total_tx_frames,
                            infer_round,
                            tx_interval_ms,
                            [round(float(v), 6) for v in tx_values],
                        )
                    )
                last_tx_us = tx_now_us

            if debug_predict_trace:
                online_print(
                    "uart_online_result: infer_round={}, preds={}, infer_costs_ms={}, full_gas_alarm={}, alarm_reason={}".format(
                        infer_round,
                        [round(float(v), 6) for v in model_values],
                        [round(float(v), 3) for v in infer_costs_ms],
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                        full_gas_alarm.last_reason,
                    )
                )

            if infer_round % 20 == 0:
                online_print(
                    "uart_online_stat: rx_frames={}, tx_frames={}, base_window_count={}, infer_round={}, last_preds={}, full_gas_alarm={}".format(
                        total_rx_frames,
                        total_tx_frames,
                        base_window_count,
                        infer_round,
                        [round(float(ctx.last_pred), 6) for ctx in model_contexts],
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                    )
                )
                gc.collect()


def run_multi_csv_cached(cfg, root):
    """
    中文注释：多模型 CSV 批量推理模式。

    这里采用“统一样本，再各模型裁剪”的策略：
    1. 统一按最大 sequence_length 生成共享样本。
    2. CNN 只取共享样本的最后 1 窗。
    3. CNN-TCN / CNN-LSTM 取完整序列。
    4. 最终输出一张合并后的预测 CSV。
    """
    runtime_cfg = cfg.get("runtime", {})
    csv_cfg = base.get_runtime_section(runtime_cfg, "csv_cached")
    model_cfgs = cfg.get("models", [])
    model_contexts = [ModelRuntimeContext(root, item) for item in model_cfgs]
    validate_multi_models(model_contexts)

    paths_cfg = require_field(cfg, "paths", "config")
    pred_csv = base.join_path(root, require_field(paths_cfg, "predictions_csv", "config.paths"))

    max_samples = csv_cfg.get("max_samples", runtime_cfg.get("max_samples", None))
    if max_samples is not None:
        max_samples = base.require_positive_int(max_samples, "runtime.csv_cached.max_samples")

    infer_batch_size = int(csv_cfg.get("infer_batch_size", 12))
    if infer_batch_size <= 0:
        infer_batch_size = 1

    write_csv = bool(csv_cfg.get("write_predictions_csv", True))

    t_start = base.now_us()
    X_shared, y_all, rebuilt = ensure_shared_dataset_cache(cfg, root, model_contexts, max_samples)
    if rebuilt:
        print("dataset_cache_rebuilt_samples:", int(X_shared.shape[0]))
    else:
        print("dataset_cache_hit_samples:", int(X_shared.shape[0]))

    start_idx, count = acquire_multi_infer_range(int(X_shared.shape[0]), infer_batch_size)
    y_batch = collect_labels_range(y_all, start_idx, count)
    X_batch = collect_shared_sample_range(X_shared, start_idx, count)

    pred_map = {}
    infer_us_map = {}
    for ctx in model_contexts:
        preds = []
        infer_us_total = 0
        scaled_sample = base.empty_float((ctx.sequence_length, ctx.window_size))
        for i in range(count):
            model_sample = adapt_shared_sample_for_model(ctx, X_batch[i])
            model_sample = scale_sample_for_model(ctx, model_sample, scaled_sample)
            pred, infer_us = run_prebuilt_sample(ctx, model_sample)
            preds.append(pred)
            infer_us_total += infer_us
            if (i + 1) % 64 == 0:
                gc.collect()
        pred_map[ctx.name] = base.as_float_array(preds)
        infer_us_map[ctx.name] = infer_us_total

    if write_csv:
        write_multi_predictions(pred_csv, y_batch, pred_map, [ctx.name for ctx in model_contexts])

    t_end = base.now_us()
    total_us = base.diff_us(t_end, t_start)

    print("=== K230 Multi-Model CSV Inference ===")
    print("root:", root)
    print("config_name:", cfg.get("name", ""))
    print("mode:", "csv_cached")
    print("dataset_total_samples:", int(X_shared.shape[0]))
    print("shared_input_shape:", tuple(X_shared.shape[1:]))
    print("infer_batch_size:", int(count))
    print("infer_start_idx:", int(start_idx))
    print("pipeline_total_time_sec:", total_us / 1_000_000.0)
    print("write_predictions_csv:", bool(write_csv))
    if write_csv:
        print("prediction_csv:", pred_csv)

    for ctx in model_contexts:
        preds = pred_map[ctx.name]
        infer_us = infer_us_map[ctx.name]
        print("--- model:", ctx.name, "---")
        print("kmodel:", ctx.kmodel_path)
        print("input_shape:", (ctx.sequence_length, ctx.window_size))
        print("model_infer_time_sec:", infer_us / 1_000_000.0)
        print("model_infer_time_per_sample_ms:", infer_us / 1000.0 / float(count))
        print("MAE:", base.safe_metric_mae(y_batch, preds))
        print("RMSE:", base.safe_metric_rmse(y_batch, preds))
        print("first_10_predictions:", preds[:10].tolist())


def main():
    """
    中文注释：多模型入口的主函数。

    整体上故意保持和老脚本很像，这样后续维护时切换成本更低：
    1. 自动探测根目录
    2. 读取多模型配置
    3. 初始化 UART
    4. 进入多模型在线推理循环
    """
    root = base.detect_root()
    cli_args = []
    if base.sys is not None and hasattr(base.sys, "argv"):
        cli_args = list(base.sys.argv[1:])
    config_path = resolve_multi_config_path(root, cli_args)
    cfg = base.load_json(config_path)

    runtime_cfg = cfg.get("runtime", {})
    mode = base.normalize_runtime_mode(runtime_cfg.get("mode", "uart_online"))

    if mode == "uart_online":
        uart_cfg = cfg.get("uart", {})
        uart_sender = base.UartDrynessSender(uart_cfg)
        run_multi_uart_online(
            cfg=cfg,
            root=root,
            uart_sender=uart_sender,
        )
        return

    if mode == "csv_cached":
        run_multi_csv_cached(
            cfg=cfg,
            root=root,
        )
        return

    raise ValueError("Unsupported multi-model runtime.mode: {}".format(mode))


if __name__ == "__main__":
    main()
