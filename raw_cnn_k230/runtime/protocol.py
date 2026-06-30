"""返回帧协议常量和数值打包工具。"""


RAW_ANOMALY_OK = 0x00
RAW_ANOMALY_ALL_ZERO = 0x01
RAW_ANOMALY_LOW = 0x02
RAW_ANOMALY_HIGH = 0x03
RAW_ANOMALY_SPIKE = 0x04
RAW_ANOMALY_STUCK = 0x05
FULL_GAS_ALARM_CODE = 0x10


def is_finite_number(value):
    """中文注释：只允许正常有限数参与协议打包。"""
    v = float(value)
    if v != v:
        return False
    if v == float("inf") or v == float("-inf"):
        return False
    return True


def clamp_int32(value):
    """中文注释：把整数限制到 int32 可表达范围内。"""
    if value > 2147483647:
        return 2147483647
    if value < -2147483648:
        return -2147483648
    return int(value)


def pack_alarm_dryness(error_code, dryness_value, dryness_scale=100.0):
    """中文注释：4 字节返回格式为 [异常码 1 字节][保留 1 字节][干度 uint16]。"""
    code = int(error_code) & 0xFF
    dry = 0
    if is_finite_number(dryness_value):
        dry = int(round(float(dryness_value) * float(dryness_scale)))
    if dry < 0:
        dry = 0
    if dry > 65535:
        dry = 65535
    return clamp_int32((code << 24) | int(dry))
