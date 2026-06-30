import argparse
import random
import struct
import time


INNER_HEADER = bytes([0x55, 0xAA])
INNER_TAIL = bytes([0xFC, 0xCF])
OUTER_HEADER = bytes([0xF7, 0x7F])
OUTER_TAIL = bytes([0xFA, 0xAF])


def parse_args():
    parser = argparse.ArgumentParser(description="PC side UART input simulator for K230 online inference.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--mode", choices=["normal", "all_zero", "stuck", "spike"], default="normal")
    parser.add_argument("--outer-count", type=int, default=20)
    parser.add_argument("--value-count", type=int, default=12)
    parser.add_argument("--base", type=int, default=600000)
    parser.add_argument("--jitter", type=int, default=300)
    parser.add_argument("--spike-delta", type=int, default=500000)
    parser.add_argument("--send-outer-frames", type=int, default=90, help="How many outer frames to send.")
    parser.add_argument("--interval-ms", type=float, default=10.0, help="Sleep time between outer frames.")
    parser.add_argument("--read-timeout", type=float, default=0.02)
    return parser.parse_args()


def build_inner_frame(values):
    payload = bytearray()
    for value in values:
        payload.extend(struct.pack(">i", int(value)))
    return INNER_HEADER + bytes(payload) + INNER_TAIL


def build_outer_frame(inner_frames):
    return OUTER_HEADER + b"".join(inner_frames) + OUTER_TAIL


def make_values(mode, frame_index, value_count, base, jitter, spike_delta):
    if mode == "all_zero":
        return [0] * value_count

    if mode == "stuck":
        return [base] * value_count

    values = []
    for channel in range(value_count):
        wave = ((frame_index + channel * 7) % 80) - 40
        noise = random.randint(-jitter, jitter)
        values.append(base + wave * 20 + noise)

    if mode == "spike" and frame_index % 37 == 0:
        values[0] += spike_delta

    return values


def try_read_reply(ser, value_count):
    # 中文注释：板端返回仍是 55 AA + 12 * int32 + FC CF；这里顺手解析异常码和干度。
    frame_len = len(INNER_HEADER) + value_count * 4 + len(INNER_TAIL)
    data = ser.read(frame_len)
    if len(data) != frame_len:
        return None
    if data[:2] != INNER_HEADER or data[-2:] != INNER_TAIL:
        return {"raw_hex": data.hex(" ")}

    values = []
    payload = data[2:-2]
    for i in range(value_count):
        raw_value = struct.unpack(">i", payload[i * 4 : i * 4 + 4])[0]
        error_code = (raw_value >> 24) & 0xFF
        dryness = (raw_value & 0xFFFF) / 100.0
        values.append({"raw": raw_value, "error": error_code, "dryness": dryness})
    return values


def main():
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Missing dependency: install pyserial first, for example `pip install pyserial`.") from exc

    args = parse_args()
    ser = serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.read_timeout,
        write_timeout=1.0,
    )

    print("UART simulator started:", args.port, args.baudrate, "mode=", args.mode)
    print("Outer frame: F7 7F + {} * inner frame + FA AF".format(args.outer_count))
    try:
        small_frame_index = 0
        for outer_index in range(args.send_outer_frames):
            inner_frames = []
            for _ in range(args.outer_count):
                values = make_values(
                    args.mode,
                    small_frame_index,
                    args.value_count,
                    args.base,
                    args.jitter,
                    args.spike_delta,
                )
                inner_frames.append(build_inner_frame(values))
                small_frame_index += 1

            packet = build_outer_frame(inner_frames)
            ser.write(packet)
            ser.flush()

            reply = try_read_reply(ser, args.value_count)
            if reply is not None:
                print("rx reply after outer #{:04d}: {}".format(outer_index + 1, reply[:4] if isinstance(reply, list) else reply))

            if args.interval_ms > 0:
                time.sleep(args.interval_ms / 1000.0)

            if (outer_index + 1) % 10 == 0:
                print("sent outer frames:", outer_index + 1, "small frames:", small_frame_index)
    finally:
        ser.close()
        print("UART simulator stopped.")


if __name__ == "__main__":
    main()
