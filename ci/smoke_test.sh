#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: COMFYUI_HOME=/path/to/ComfyUI ci/smoke_test.sh <workflow.json> [<workflow.json>...]" >&2
  exit 1
fi

if [[ -z "${COMFYUI_HOME:-}" ]]; then
  echo "COMFYUI_HOME environment variable must point to the ComfyUI checkout" >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
HOST=${COMFYUI_HOST:-127.0.0.1}
PORT=${COMFYUI_PORT:-8188}
LOG_FILE=${COMFYUI_LOG:-comfyui_ci.log}

pushd "${COMFYUI_HOME}" >/dev/null

${PYTHON_BIN} -m pip install --upgrade pip >/dev/null
${PYTHON_BIN} -m pip install -r requirements.txt >/dev/null

${PYTHON_BIN} main.py --disable-auto-launch --listen "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true' EXIT

popd >/dev/null

"${PYTHON_BIN}" "$(dirname "$0")/run_workflows.py" --host "${HOST}" --port "${PORT}" "$@"
