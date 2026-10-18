#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${RDN_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Set RDN_PYTHON to the Python executable in your existing CUDA environment.' >&2
  exit 1
fi
export RDN_DATA_ROOT="${RDN_DATA_ROOT:-$("$PYTHON" -c 'import json,pathlib; p=pathlib.Path(".local_config.json"); print(json.loads(p.read_text()).get("data_root", "") if p.exists() else "")')}"
if [[ -z "$RDN_DATA_ROOT" ]]; then
  echo 'Set RDN_DATA_ROOT to your existing RDN package directory.' >&2
  exit 1
fi
export RDN_PRETRAINED_DIR="${RDN_PRETRAINED_DIR:-$RDN_DATA_ROOT/pretrained}"
export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.jupyter/config"
export JUPYTER_DATA_DIR="$PROJECT_DIR/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.jupyter/runtime"
export IPYTHONDIR="$PROJECT_DIR/.ipython"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
"$PYTHON" - <<'PY'
import json, os, pathlib, sys
folder = pathlib.Path(os.environ['JUPYTER_DATA_DIR']) / 'kernels/rdn-finetune'
folder.mkdir(parents=True, exist_ok=True)
(folder/'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'RDN fine-tuning', 'language': 'python'}, indent=2) + '\n')
PY
exec "$PYTHON" -m jupyterlab --ServerApp.ip=127.0.0.1 \
  --ServerApp.root_dir="$PROJECT_DIR" --LabApp.default_url=/lab/tree/standalone_notebooks "$@"
