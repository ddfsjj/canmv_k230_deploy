"""
独立串口持续发送测试脚本。

用途：
1. 不依赖模型与 CSV，单独验证 UART 引脚、波特率和协议格式是否正确。
2. 上位机或单片机可先用这个脚本确认“发送端”本身没有问题。
3. 当前按固定协议持续发包：55 AA + 12 个 4 字节数值 + FC CF。
"""

from machine import UART, FPIOA
import time

try:
    import ustruct as struct
except ImportError:
    import struct

# =========================
# 串口物理层配置
# 与单片机约定保持一致
# =========================
UART_ID_NUM = 2
TX_PIN = 11
RX_PIN = 12
BAUDRATE = 921600
BITS = 8
PARITY = "none"  # none / even / odd
STOP = 1

# =========================
# 帧协议配置
# 帧格式：
# 55 AA + 12 个 4 字节数值（高位在前）+ FC CF
# =========================
FRAME_HEADER = b"\x55\xAA"
FRAME_TAIL = b"\xFC\xCF"
VALUE_COUNT = 12
SEND_INTERVAL_MS = 500
PRINT_FRAME_HEX = True


def _now_ms():
    # 兼容不同运行时的毫秒接口。
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def _sleep_ms(ms):
    # 兼容不同运行时的休眠接口。
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


def _pack_u32_be(v):
    # 4 字节无符号整型，大端序（高位在前）。
    # 即十六进制显示顺序与人工阅读顺序一致。
    return struct.pack(">I", int(v) & 0xFFFFFFFF)


def _build_frame(values):
    # 把输入值打成一整帧协议数据。
    # 不足 12 路时自动补 0，保证帧长固定。
    frame = bytearray(FRAME_HEADER)
    for i in range(VALUE_COUNT):
        v = 0
        if i < len(values):
            v = values[i]
        frame.extend(_pack_u32_be(v))
    frame.extend(FRAME_TAIL)
    return frame


def _bytes_to_hex(data):
    # 仅用于打印调试，便于直接和串口助手看到的十六进制内容比对。
    return " ".join("{:02X}".format(b) for b in data)


def _uart_consts():
    # 把用户填写的整数/字符串配置转换成 machine.UART 所需常量。
    uart_const = getattr(UART, "UART{}".format(UART_ID_NUM), UART_ID_NUM)
    bits_const = UART.SEVENBITS if int(BITS) == 7 else UART.EIGHTBITS
    parity_key = str(PARITY).lower()
    if parity_key == "even":
        parity_const = UART.PARITY_EVEN
    elif parity_key == "odd":
        parity_const = UART.PARITY_ODD
    else:
        parity_const = UART.PARITY_NONE
    stop_const = UART.STOPBITS_TWO if int(STOP) == 2 else UART.STOPBITS_ONE
    return uart_const, bits_const, parity_const, stop_const


def _setup_fpioa():
    # 将物理引脚映射为 UART 功能。
    # 这里是 K230 上 UART2 的 TX/RX 复用关系。
    fpioa = FPIOA()
    tx_func = getattr(fpioa, "UART{}_TXD".format(UART_ID_NUM))
    rx_func = getattr(fpioa, "UART{}_RXD".format(UART_ID_NUM))
    try:
        fpioa.set_function(TX_PIN, tx_func, ie=1, oe=1)
    except TypeError:
        fpioa.set_function(TX_PIN, tx_func)
    try:
        fpioa.set_function(RX_PIN, rx_func, ie=1, oe=1)
    except TypeError:
        fpioa.set_function(RX_PIN, rx_func)


def main():
    # 先完成引脚复用，再初始化 UART 外设。
    _setup_fpioa()
    uart_const, bits_const, parity_const, stop_const = _uart_consts()
    uart = UART(
        uart_const,
        baudrate=BAUDRATE,
        bits=bits_const,
        parity=parity_const,
        stop=stop_const,
    )

    frame_len = len(FRAME_HEADER) + VALUE_COUNT * 4 + len(FRAME_TAIL)
    print("UART持续发送测试开始")
    print("UART{} {}bps TX={} RX={} 8N1".format(UART_ID_NUM, BAUDRATE, TX_PIN, RX_PIN))
    print("Frame header=55 AA, tail=FC CF, value_count={}, len={}".format(VALUE_COUNT, frame_len))
    print("按 Ctrl+C 停止")

    seq = 0
    try:
        while True:
            # 当前测试包内容：
            # 第 1 个数值：递增序号，便于确认包序是否连续。
            # 第 2 个数值：毫秒计时，便于确认是否一直在发。
            # 其余数值补 0，只占位，不代表业务数据。
            values = [seq, _now_ms()]
            frame = _build_frame(values)
            written = uart.write(frame)
            if PRINT_FRAME_HEX:
                print("TX seq={} bytes={} hex={}".format(seq, written, _bytes_to_hex(frame)))
            else:
                print("TX seq={} bytes={}".format(seq, written))
            seq += 1
            _sleep_ms(SEND_INTERVAL_MS)
    except KeyboardInterrupt:
        print("收到停止信号，结束测试")
    finally:
        try:
            uart.deinit()
        except Exception:
            pass
        print("UART已释放")


if __name__ == "__main__":
    main()
