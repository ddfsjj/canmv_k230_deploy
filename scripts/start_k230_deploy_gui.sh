#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${K230_DEPLOY_PID_FILE:-$ROOT_DIR/.k230_deploy_gui.pid}"
PORT_FILE="${K230_DEPLOY_PORT_FILE:-$ROOT_DIR/.k230_deploy_gui.port}"
LOG_FILE="${K230_DEPLOY_LOG_FILE:-$ROOT_DIR/k230_deploy_gui.log}"
BASE_PORT="${K230_DEPLOY_PORT:-7861}"
PORT_RANGE="${K230_DEPLOY_PORT_RANGE:-20}"
VQ_K230_OUTPUT_DIR="${VQ_K230_OUTPUT_DIR:-/opt/vq/VQ_Estimator/outputs/k230_quant}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$CONDA_BASE" && -x "$CONDA_BASE/envs/vq/bin/python" ]]; then
      PYTHON_BIN="$CONDA_BASE/envs/vq/bin/python"
    else
      PYTHON_BIN="$(command -v python3)"
    fi
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

process_cmdline() {
  local pid="$1"
  if [[ -r "/proc/$pid/cmdline" ]]; then
    tr '\0' ' ' <"/proc/$pid/cmdline"
  else
    ps -p "$pid" -o args= 2>/dev/null || true
  fi
}

process_cwd() {
  local pid="$1"
  if [[ -L "/proc/$pid/cwd" ]]; then
    readlink -f "/proc/$pid/cwd" 2>/dev/null || true
  fi
}

is_k230_gui_process() {
  local pid="$1"
  local cmd cwd
  cmd="$(process_cmdline "$pid")"
  cwd="$(process_cwd "$pid")"
  if [[ "$cmd" != *"vq_deploy_gradio.py"* && "$cmd" != *"k230_deploy_gui.py"* ]]; then
    return 1
  fi
  [[ "$cwd" == "$ROOT_DIR" || "$cmd" == *"$ROOT_DIR"* ]]
}

pid_for_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1
  fi
}

port_for_pid() {
  local pid="$1"
  local start_port="$BASE_PORT"
  local end_port=$((BASE_PORT + PORT_RANGE - 1))
  local port owner
  for port in $(seq "$start_port" "$end_port"); do
    owner="$(pid_for_port "$port" || true)"
    if [[ "$owner" == "$pid" ]]; then
      echo "$port"
      return 0
    fi
  done
}

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1 && is_k230_gui_process "$pid"
}

find_existing_k230() {
  local start_port="$BASE_PORT"
  local end_port=$((BASE_PORT + PORT_RANGE - 1))
  local port pid
  for port in $(seq "$start_port" "$end_port"); do
    pid="$(pid_for_port "$port" || true)"
    if [[ -n "$pid" ]] && is_k230_gui_process "$pid"; then
      echo "$pid $port"
      return 0
    fi
  done
}

find_free_port() {
  local start_port="$BASE_PORT"
  local end_port=$((BASE_PORT + PORT_RANGE - 1))
  local port pid
  for port in $(seq "$start_port" "$end_port"); do
    pid="$(pid_for_port "$port" || true)"
    if [[ -z "$pid" ]]; then
      echo "$port"
      return 0
    fi
  done
  return 1
}

print_urls() {
  local port="$1"
  echo "URL:"
  echo "  local:   http://127.0.0.1:$port"
  if command -v hostname >/dev/null 2>&1; then
    local printed=0
    for ip in $(hostname -I 2>/dev/null || true); do
      if [[ -n "$ip" ]]; then
        echo "  server:  http://$ip:$port"
        printed=1
      fi
    done
    if [[ "$printed" == "0" ]]; then
      echo "  server:  http://<server-ip>:$port"
    fi
  else
    echo "  server:  http://<server-ip>:$port"
  fi
}

start() {
  local pid port existing selected owner

  if is_running; then
    pid="$(cat "$PID_FILE")"
    port="$(port_for_pid "$pid" || true)"
    if [[ -z "$port" && -f "$PORT_FILE" ]]; then
      port="$(cat "$PORT_FILE" 2>/dev/null || true)"
    fi
    echo "K230 deploy GUI is already running. pid=$pid port=${port:-unknown}"
    echo "python: $(process_cmdline "$pid")"
    if [[ -n "$port" ]]; then
      print_urls "$port"
    fi
    return 0
  fi

  existing="$(find_existing_k230 || true)"
  if [[ -n "$existing" ]]; then
    pid="${existing%% *}"
    port="${existing##* }"
    echo "$pid" >"$PID_FILE"
    echo "$port" >"$PORT_FILE"
    echo "K230 deploy GUI is already running. pid=$pid port=$port"
    echo "python: $(process_cmdline "$pid")"
    print_urls "$port"
    return 0
  fi

  selected="$(find_free_port || true)"
  if [[ -z "$selected" ]]; then
    echo "Cannot find empty port in range: $BASE_PORT-$((BASE_PORT + PORT_RANGE - 1))"
    echo "Set a wider range with: K230_DEPLOY_PORT_RANGE=50 ./scripts/start_gui.sh start"
    return 1
  fi

  mkdir -p "$(dirname "$PID_FILE")"
  cd "$ROOT_DIR"
  export VQ_K230_OUTPUT_DIR
  export K230_DEPLOY_PORT="$selected"
  export K230_DEPLOY_PORT_RANGE="1"
  export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

  nohup "$PYTHON_BIN" scripts/vq_deploy_gradio.py >>"$LOG_FILE" 2>&1 &
  pid="$!"
  echo "$pid" >"$PID_FILE"
  echo "$selected" >"$PORT_FILE"

  for _ in {1..30}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "K230 deploy GUI failed to start. See log: $LOG_FILE"
      rm -f "$PID_FILE" "$PORT_FILE"
      return 1
    fi
    owner="$(pid_for_port "$selected" || true)"
    if [[ "$owner" == "$pid" ]]; then
      echo "K230 deploy GUI started. pid=$pid port=$selected"
      echo "python: $PYTHON_BIN"
      echo "log: $LOG_FILE"
      print_urls "$selected"
      return 0
    fi
    sleep 0.2
  done

  echo "K230 deploy GUI did not open port $selected in time. See log: $LOG_FILE"
  rm -f "$PID_FILE" "$PORT_FILE"
  return 1
}

stop_pid() {
  local pid="$1"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
}

stop() {
  local pid existing port
  if is_running; then
    pid="$(cat "$PID_FILE")"
    stop_pid "$pid"
    echo "K230 deploy GUI stopped. pid=$pid"
  else
    echo "K230 deploy GUI is not running."
  fi
  rm -f "$PID_FILE" "$PORT_FILE"

  while true; do
    existing="$(find_existing_k230 || true)"
    if [[ -z "$existing" ]]; then
      break
    fi
    pid="${existing%% *}"
    port="${existing##* }"
    echo "Stopping orphan K230 deploy GUI process on port $port. pid=$pid"
    stop_pid "$pid"
  done
}

status() {
  local pid port existing
  if is_running; then
    pid="$(cat "$PID_FILE")"
    port="$(port_for_pid "$pid" || true)"
    if [[ -z "$port" && -f "$PORT_FILE" ]]; then
      port="$(cat "$PORT_FILE" 2>/dev/null || true)"
    fi
    echo "K230 deploy GUI is running. pid=$pid port=${port:-unknown}"
    echo "python: $(process_cmdline "$pid")"
    echo "VQ_K230_OUTPUT_DIR=$VQ_K230_OUTPUT_DIR"
    echo "log: $LOG_FILE"
    if [[ -n "$port" ]]; then
      print_urls "$port"
    fi
    return 0
  fi

  existing="$(find_existing_k230 || true)"
  if [[ -n "$existing" ]]; then
    pid="${existing%% *}"
    port="${existing##* }"
    echo "$pid" >"$PID_FILE"
    echo "$port" >"$PORT_FILE"
    echo "K230 deploy GUI is running. pid=$pid port=$port"
    echo "python: $(process_cmdline "$pid")"
    echo "VQ_K230_OUTPUT_DIR=$VQ_K230_OUTPUT_DIR"
    echo "log: $LOG_FILE"
    print_urls "$port"
  else
    echo "K230 deploy GUI is not running."
  fi
}

case "${1:-}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 2
    ;;
esac
