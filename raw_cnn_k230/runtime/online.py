"""UART 在线运行主循环。

这里负责把输入层、窗口层、推理绑定、状态层和输出层串起来。
旧的 run_k230_multi_infer.py 只保留兼容入口，真正在线运行路径放到 runtime 包内。
"""

import gc

from runtime import bindings as runtime_bindings
from runtime import config as runtime_config
from runtime import guards as runtime_guards
from runtime import inputs as runtime_inputs
from runtime import numeric
from runtime import output as runtime_output
from runtime import platform
from runtime import status as runtime_status
from runtime import windows as runtime_windows


def send_zero_frame(uart_sender):
    """中文注释：未就绪时按当前协议发送一帧全零，方便上位机保持节拍。"""
    uart_sender.send_values_frame([0.0] * int(uart_sender.value_count))


def run_uart_online(cfg, root, uart_sender):
    """中文注释：统一 UART 在线运行主循环，单模型和多模型都通过 bindings 表达。"""
    runtime_cfg = cfg.get("runtime", {})
    online_cfg = runtime_config.get_runtime_section(runtime_cfg, "uart_online")
    output_cfg = runtime_config.get_runtime_section(runtime_cfg, "output")
    model_cfgs = cfg.get("models", [])
    model_contexts = [runtime_bindings.ModelRuntimeContext(root, item) for item in model_cfgs]
    runtime_bindings.validate_multi_models(model_contexts)

    if uart_sender is None or not uart_sender.enabled or uart_sender.uart is None:
        raise RuntimeError("UART sender is disabled; uart_online mode cannot start.")

    common = model_contexts[0]
    window_size = common.window_size
    base_step = common.base_step
    feature_mode = common.feature_mode
    max_seq_length = runtime_bindings.get_common_runtime_shape(model_contexts)[3]

    channel_count = runtime_config.require_positive_int(
        online_cfg.get("channel_count", 1),
        "runtime.uart_online.channel_count",
    )
    input_contexts = runtime_bindings.parse_multi_inputs(cfg, channel_count)
    model_bindings = runtime_bindings.make_model_input_bindings(model_contexts, model_cfgs, input_contexts)
    status_ctx = runtime_status.RuntimeStatusContext(
        cfg=cfg,
        runtime_cfg=runtime_cfg,
        channel_count=channel_count,
        binding_count=len(model_bindings),
    )
    zero_guard_cfg = status_ctx.zero_guard_cfg
    zero_guard_enabled = status_ctx.zero_guard_enabled
    zero_guard_output_value = status_ctx.zero_guard_output_value
    raw_anomaly = status_ctx.raw_anomaly
    channel_raw_error_codes = status_ctx.channel_raw_error_codes
    postprocessor = status_ctx.postprocessor
    full_gas_alarm = status_ctx.full_gas_alarm

    output_source_by_name = {}
    for binding in model_bindings:
        output_source_by_name[binding.output_name] = int(binding.input_ctx["source_index"])

    input_index_by_name = {}
    for idx, item in enumerate(input_contexts):
        input_index_by_name[item["name"]] = idx

    idle_sleep_ms = int(online_cfg.get("idle_sleep_ms", 1))
    warmup_send = bool(online_cfg.get("send_zeros_before_ready", False))
    quiet = bool(online_cfg.get("quiet", False))
    debug_predict_trace = bool(online_cfg.get("debug_predict_trace", True))
    debug_tx_timing = bool(online_cfg.get("debug_tx_timing", False))
    debug_tx_only_abnormal = bool(online_cfg.get("debug_tx_only_abnormal", True))
    debug_tx_interval_min_warn_ms = float(online_cfg.get("debug_tx_interval_min_warn_ms", 180.0))
    debug_tx_interval_max_warn_ms = float(online_cfg.get("debug_tx_interval_max_warn_ms", 240.0))
    flush_rx_on_start = bool(online_cfg.get("flush_rx_on_start", True))
    startup_flush_empty_rounds = int(online_cfg.get("startup_flush_empty_rounds", 3))
    startup_flush_sleep_ms = int(online_cfg.get("startup_flush_sleep_ms", 10))

    total_rx_frames = 0
    total_tx_frames = 0
    infer_round = 0

    window_bank = runtime_windows.OnlineWindowBank(
        input_contexts=input_contexts,
        window_size=window_size,
        base_step=base_step,
        feature_mode=feature_mode,
        max_seq_length=max_seq_length,
        zero_guard_enabled=zero_guard_enabled,
    )
    tmp_seq_map = {}
    tmp_sample_map = {}
    zero_guard_state_map = {}
    for binding in model_bindings:
        zero_guard_state_map[binding.output_name] = runtime_guards.ZeroGuardState(zero_guard_cfg)
        if binding.model_ctx.sequence_length > 1:
            tmp_seq_map[binding.output_name] = numeric.empty_float((binding.model_ctx.sequence_length, window_size))
            tmp_sample_map[binding.output_name] = numeric.empty_float((1, binding.model_ctx.sequence_length, window_size))
        else:
            tmp_seq_map[binding.output_name] = None
            tmp_sample_map[binding.output_name] = numeric.empty_float((1, 1, window_size))

    def online_print(*args):
        if not quiet:
            print(*args)

    input_reader = runtime_inputs.UartOnlineInputReader(
        uart_sender=uart_sender,
        channel_count=channel_count,
        online_cfg=online_cfg,
        online_print=online_print,
        quiet=quiet,
    )

    online_print("=== K230 Unified UART Runtime ===")
    online_print("config_name:", cfg.get("name", ""))
    online_print("mode:", "uart_online")
    online_print("model_count:", len(model_contexts))
    online_print("binding_count:", len(model_bindings))
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
    online_print("postprocessing:", status_ctx.postprocessing_summary())
    online_print("raw_anomaly:", status_ctx.raw_anomaly_summary())
    online_print(
        "uart_online_cfg: channels={}, window={}, base_step={}, feature_mode={}, output_value_count={}".format(
            channel_count,
            window_size,
            base_step,
            feature_mode,
            uart_sender.value_count,
        )
    )
    online_print("uart_online_zero_guard:", status_ctx.zero_guard_summary())
    online_print("uart_online_full_gas_alarm:", status_ctx.full_gas_summary())

    if flush_rx_on_start:
        flushed_bytes = platform.drain_uart_rx(
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

    session_start_us = platform.now_us()
    first_rx_us = None
    last_infer_trigger_us = None
    last_tx_us = None

    while True:
        read_batch = input_reader.read_batch(total_rx_frames)
        if read_batch is None:
            platform.sleep_ms(idle_sleep_ms)
            continue
        if not read_batch.frames:
            continue

        for values in read_batch.frames:
            total_rx_frames += 1
            if first_rx_us is None:
                first_rx_us = platform.now_us()

            if not window_bank.push_values(values, status_ctx):
                if warmup_send:
                    send_zero_frame(uart_sender)
                    total_tx_frames += 1
                continue

            freq_mean = window_bank.last_freq_mean
            first_base_window = window_bank.last_first_base_window

            ready_count = 0
            for binding in model_bindings:
                binding.update_with_base_window(window_bank.get_feature_window(binding.input_ctx["name"]))
                if binding.ready:
                    ready_count += 1

            if ready_count != len(model_bindings):
                if warmup_send:
                    send_zero_frame(uart_sender)
                    total_tx_frames += 1
                continue

            if debug_predict_trace:
                trigger_now_us = platform.now_us()
                elapsed_from_start_ms = platform.diff_us(trigger_now_us, session_start_us) / 1000.0
                elapsed_from_first_rx_ms = -1.0
                if first_rx_us is not None:
                    elapsed_from_first_rx_ms = platform.diff_us(trigger_now_us, first_rx_us) / 1000.0
                since_last_infer_ms = -1.0
                if last_infer_trigger_us is not None:
                    since_last_infer_ms = platform.diff_us(trigger_now_us, last_infer_trigger_us) / 1000.0
                last_infer_trigger_us = trigger_now_us
                online_print(
                    "uart_online_trigger: infer_round_next={}, rx_small_frame_idx={}, base_window_idx={}, first_base_window={}, ready_models={}, elapsed_start_ms={:.3f}, elapsed_first_rx_ms={:.3f}, since_last_infer_ms={:.3f}".format(
                        infer_round + 1,
                        total_rx_frames,
                        window_bank.base_window_count,
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
                if zero_guard_enabled and window_bank.zero_seq_filled >= max_seq_length:
                    input_idx = input_index_by_name[binding.input_ctx["name"]]
                    tmp_zero_seq = window_bank.expand_zero_guard_sequence(input_idx)
                    guard_scaled_seq = None
                    if binding.model_ctx.sequence_length > 1:
                        numeric.expand_sequence_ring(
                            binding.seq_ring,
                            binding.seq_write_idx,
                            tmp_seq_map[binding.output_name],
                        )
                        guard_scaled_seq = tmp_seq_map[binding.output_name]
                    binding_zero_guard_hit, zero_guard_votes, zero_guard_features = runtime_status.check_zero_guard(
                        tmp_zero_seq,
                        guard_scaled_seq,
                        zero_guard_cfg,
                        state=zero_guard_state_map[binding.output_name],
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
                                runtime_status.format_zero_guard_features(zero_guard_features),
                            )
                        )
                else:
                    pred, infer_us = binding.run_inference(
                        window_bank.get_feature_window(binding.input_ctx["name"]),
                        tmp_seq_map[binding.output_name],
                        tmp_sample_map[binding.output_name],
                    )
                    pred = postprocessor.update(binding_idx, pred)
                    binding.last_pred = pred
                    binding.model_ctx.last_pred = pred
                model_values.append(pred)
                model_pred_map[binding.output_name] = pred
                infer_costs_ms.append(infer_us / 1000.0)

            status_ctx.update_full_gas_alarm(
                model_values,
                freq_mean,
                zero_guard_hit=zero_guard_hit,
            )

            tx_values, tx_error_codes = runtime_output.send_uart_prediction_frame(
                uart_sender=uart_sender,
                model_pred_map=model_pred_map,
                output_cfg=output_cfg,
                output_source_by_name=output_source_by_name,
                channel_raw_error_codes=channel_raw_error_codes,
                raw_anomaly_enabled=bool(raw_anomaly.enabled),
                full_gas_alarm=full_gas_alarm,
            )
            total_tx_frames += 1

            if debug_tx_timing:
                tx_now_us = platform.now_us()
                tx_interval_ms = -1.0
                if last_tx_us is not None:
                    tx_interval_ms = platform.diff_us(tx_now_us, last_tx_us) / 1000.0
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
                    "uart_online_result: infer_round={}, preds={}, infer_costs_ms={}, raw_error_codes={}, full_gas_alarm={}, alarm_reason={}".format(
                        infer_round,
                        [round(float(v), 6) for v in model_values],
                        [round(float(v), 3) for v in infer_costs_ms],
                        channel_raw_error_codes,
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                        full_gas_alarm.last_reason,
                    )
                )

            if infer_round % 20 == 0:
                online_print(
                    "uart_online_stat: rx_frames={}, tx_frames={}, base_window_count={}, infer_round={}, last_preds={}, raw_error_codes={}, full_gas_alarm={}".format(
                        total_rx_frames,
                        total_tx_frames,
                        window_bank.base_window_count,
                        infer_round,
                        [round(float(ctx.last_pred), 6) for ctx in model_contexts],
                        channel_raw_error_codes,
                        bool(full_gas_alarm.alarm_on) if full_gas_alarm.enabled else False,
                    )
                )
                gc.collect()
