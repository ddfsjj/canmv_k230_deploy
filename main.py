import sys
import time

APP_DIR = "/sdcard/raw_cnn_k230"
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def run_once():
    import run_k230_infer as infer_app

    infer_app.main()


while True:
    try:
        run_once()
    except Exception as exc:
        print("UART auto-start error:", exc)
        if hasattr(time, "sleep_ms"):
            time.sleep_ms(1000)
        else:
            time.sleep(1)
