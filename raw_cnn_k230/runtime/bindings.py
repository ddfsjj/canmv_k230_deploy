"""模型绑定层。

这里集中处理：
1. 模型运行时上下文
2. 输入通道展开
3. 模型 + 输入路绑定
4. CSV 共享样本裁剪和单模型推理
"""

from runtime import config as runtime_config
from runtime import features
from runtime import numeric
from runtime import platform

try:
    import nncase_runtime as nn  # type: ignore
except ImportError:
    nn = None  # type: ignore


def enabled_models(cfg):
    """中文注释：返回启用的模型配置。"""
    return [item for item in cfg.get("models", []) if bool(item.get("enabled", True))]


def require_field(obj, field_name, context_name):
    """中文注释：读取必填字段，缺失时直接报清晰错误。"""
    if field_name not in obj:
        raise ValueError("{} missing required field '{}'".format(context_name, field_name))
    return obj[field_name]


class ModelRuntimeContext:
    """中文注释：单个模型的运行时上下文，持有 kmodel、scaler 和 KPU 实例。"""

    def __init__(self, root, model_cfg):
        self.root = root
        self.name = str(require_field(model_cfg, "name", "model")).strip()
        self.model_type = runtime_config.normalize_model_type(require_field(model_cfg, "type", "model"))
        if self.model_type not in {"cnn", "cnn_tcn", "cnn_lstm"}:
            raise ValueError("Unsupported multi-model type: {}".format(self.model_type))

        data_cfg = require_field(model_cfg, "data", "model")
        paths_cfg = require_field(model_cfg, "paths", "model")
        preprocessing_cfg = model_cfg.get("preprocessing", {})

        self.window_size = runtime_config.require_positive_int(
            require_field(data_cfg, "base_window_size", "model.data"),
            "{}.data.base_window_size".format(self.name),
        )
        self.base_step = runtime_config.resolve_positive_step(
            data_cfg.get("base_step", None),
            self.window_size // 2,
            "{}.data.base_step".format(self.name),
        )
        self.sequence_length = runtime_config.require_positive_int(
            require_field(data_cfg, "sequence_length", "model.data"),
            "{}.data.sequence_length".format(self.name),
        )
        self.sequence_step = runtime_config.require_positive_int(
            data_cfg.get("sequence_step", 1),
            "{}.data.sequence_step".format(self.name),
        )
        self.feature_mode = runtime_config.normalize_feature_mode(preprocessing_cfg.get("feature_mode", "raw"))

        self.kmodel_path = platform.join_path(root, require_field(paths_cfg, "kmodel", "model.paths"))
        self.scaler_json_path = platform.join_path(root, require_field(paths_cfg, "scaler_json", "model.paths"))

        self.mean, self.scale = features.load_scaler_params(self.scaler_json_path)
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

        self.seq_ring = None
        self.seq_write_idx = 0
        self.seq_filled = 0
        self.seq_windows_since_infer = 0
        if self.sequence_length > 1:
            self.seq_ring = numeric.empty_float((self.sequence_length, self.window_size))

        self.last_pred = 0.0
        self.infer_count = 0
        self.total_infer_us = 0
        self.ready = False
        self.just_became_ready = False

    def update_with_base_window(self, proc_window, tmp_seq):
        """中文注释：更新模型自己的序列状态，保留旧后端兼容接口。"""
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
            numeric.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)

    def build_input_sample(self, proc_window, tmp_seq, sample3d):
        """中文注释：把当前窗口组织成 KPU 所需的三维输入。"""
        if self.sequence_length <= 1:
            sample3d.reshape((1, 1, self.window_size))[0, 0, :] = (proc_window - self.mean) / self.scale
            return sample3d.reshape((1, 1, self.window_size))

        numeric.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)
        sample3d.reshape((1, self.sequence_length, self.window_size))[0, :, :] = tmp_seq
        return sample3d.reshape((1, self.sequence_length, self.window_size))

    def run_inference(self, proc_window, tmp_seq, sample3d):
        """中文注释：执行一次模型推理。"""
        sample = self.build_input_sample(proc_window, tmp_seq, sample3d)
        input_tensor = nn.from_numpy(numeric.astype_float_array(sample))
        self.kpu.set_input_tensor(0, input_tensor)
        t0 = platform.now_us()
        self.kpu.run()
        t1 = platform.now_us()
        output = self.kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        self.last_pred = pred
        self.infer_count += 1
        self.total_infer_us += platform.diff_us(t1, t0)
        del output
        del input_tensor
        return pred, platform.diff_us(t1, t0)


class ModelInputRuntimeContext:
    """中文注释：单个“模型 + 输入路”的在线运行状态。"""

    def __init__(self, model_ctx, input_ctx, output_name):
        self.model_ctx = model_ctx
        self.input_ctx = input_ctx
        self.output_name = str(output_name)
        self.seq_ring = None
        self.seq_write_idx = 0
        self.seq_filled = 0
        self.seq_windows_since_infer = 0
        if self.model_ctx.sequence_length > 1:
            self.seq_ring = numeric.empty_float((self.model_ctx.sequence_length, self.model_ctx.window_size))
        self.ready = False
        self.just_became_ready = False
        self.last_pred = 0.0
        self.infer_count = 0
        self.total_infer_us = 0

    def update_with_base_window(self, proc_window):
        """中文注释：每一路输入独立维护序列状态。"""
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
        """中文注释：按当前绑定构造 KPU 输入样本。"""
        if self.model_ctx.sequence_length <= 1:
            sample3d.reshape((1, 1, self.model_ctx.window_size))[0, 0, :] = (
                proc_window - self.model_ctx.mean
            ) / self.model_ctx.scale
            return sample3d.reshape((1, 1, self.model_ctx.window_size))

        numeric.expand_sequence_ring(self.seq_ring, self.seq_write_idx, tmp_seq)
        sample3d.reshape((1, self.model_ctx.sequence_length, self.model_ctx.window_size))[0, :, :] = tmp_seq
        return sample3d.reshape((1, self.model_ctx.sequence_length, self.model_ctx.window_size))

    def run_inference(self, proc_window, tmp_seq, sample3d):
        """中文注释：执行当前绑定的一次推理。"""
        sample = self.build_input_sample(proc_window, tmp_seq, sample3d)
        input_tensor = nn.from_numpy(numeric.astype_float_array(sample))
        self.model_ctx.kpu.set_input_tensor(0, input_tensor)
        t0 = platform.now_us()
        self.model_ctx.kpu.run()
        t1 = platform.now_us()
        output = self.model_ctx.kpu.get_output_tensor(0)
        pred = float(output.to_numpy().reshape(-1)[0])
        infer_us = platform.diff_us(t1, t0)
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
    """中文注释：解析有效输入路配置。"""
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
    """中文注释：兼容字符串和字符串列表两种配置写法。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if str(v).strip()]
    text = str(raw_value).strip()
    if not text:
        return []
    return [text]


def make_model_input_bindings(model_contexts, model_cfgs, input_contexts):
    """中文注释：把 models 和 inputs 展开成“模型 + 输入路”的运行项。"""
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


def validate_multi_models(model_contexts):
    """中文注释：限制多模型当前必须共享窗口长度、步长和特征模式。"""
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


def get_common_runtime_shape(model_contexts):
    """中文注释：返回公共基础窗参数和最大序列长度。"""
    first = model_contexts[0]
    max_seq_length = 1
    for ctx in model_contexts:
        if ctx.sequence_length > max_seq_length:
            max_seq_length = ctx.sequence_length
    return first.window_size, first.base_step, first.feature_mode, max_seq_length


def adapt_shared_sample_for_model(model_ctx, shared_sample):
    """中文注释：把共享样本裁剪成当前模型需要的序列长度。"""
    if model_ctx.sequence_length <= 1:
        return shared_sample[-1:, :]
    return shared_sample[-model_ctx.sequence_length :, :]


def scale_sample_for_model(model_ctx, model_sample, out_sample):
    """中文注释：csv_cached 共享样本推理前按当前模型 scaler 标准化。"""
    out_sample[:] = (model_sample - model_ctx.mean) / model_ctx.scale
    return out_sample


def run_prebuilt_sample(model_ctx, sample):
    """中文注释：给 csv_cached 使用的单模型推理入口。"""
    sample3d = numeric.astype_float_array(sample).reshape((1, sample.shape[0], sample.shape[1]))
    input_tensor = nn.from_numpy(sample3d)
    model_ctx.kpu.set_input_tensor(0, input_tensor)
    t0 = platform.now_us()
    model_ctx.kpu.run()
    t1 = platform.now_us()
    output = model_ctx.kpu.get_output_tensor(0)
    pred = float(output.to_numpy().reshape(-1)[0])
    model_ctx.last_pred = pred
    model_ctx.infer_count += 1
    model_ctx.total_infer_us += platform.diff_us(t1, t0)
    del output
    del input_tensor
    return pred, platform.diff_us(t1, t0)
