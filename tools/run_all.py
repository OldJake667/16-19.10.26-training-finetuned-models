"""Run pretrained models sequentially. Fails immediately if any subprocess fails."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_code.experiment import DEFAULT_MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('RDN_DATA_ROOT'), required=not bool(os.environ.get('RDN_DATA_ROOT')))
    parser.add_argument('--models', nargs='+', choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument('--no-test', action='store_true')
    parser.add_argument('--weights-dir')
    args = parser.parse_args()
    for model in args.models:
        command = [sys.executable, str(ROOT / 'train.py'), '--model', model, '--data-root', args.data_root]
        if args.no_test:
            command.append('--no-test')
        if args.weights_dir:
            command.extend(['--weights-dir', args.weights_dir])
        subprocess.run(command, cwd=ROOT, check=True)
