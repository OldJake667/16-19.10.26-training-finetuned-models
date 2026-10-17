"""Generate clean, unexecuted notebooks for each fine-tuning model."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_code.experiment import DEFAULT_MODELS


def code(source, role):
    c = nbf.v4.new_code_cell(source.strip() + '\n')
    c.metadata['lab_role'] = role
    return c


def write(path, cells, key=None):
    n = nbf.v4.new_notebook(cells=cells)
    n.metadata.update(kernelspec={'display_name': 'RDN fine-tuning', 'language': 'python', 'name': 'rdn-finetune'},
                      language_info={'name': 'python'}, experiment={'model': key, 'training_executed': False})
    nbf.validate(n); nbf.write(n, path)


imports = '''import os
import sys
from pathlib import Path
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'lab_code/experiment.py').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Start Jupyter inside the fine-tuning repository.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab_code.experiment import get_config, run_experiment
'''

if __name__ == '__main__':
    folder = ROOT / 'notebooks'; folder.mkdir(exist_ok=True)
    for i, key in enumerate((*DEFAULT_MODELS, 'cnn'), 1):
        continuation = key == 'cnn'
        header = ('Original custom CNN: optional checkpoint continuation' if continuation else key + ': published pretrained weights, all layers fine-tuned')
        intro = ('This custom architecture has no published ImageNet checkpoint. Supply the original '
                 'best_checkpoint.pt. This is continued training, not a pretrained-vs-scratch comparison.' if continuation else
                 'Uses the same full compact RDN data and fixed split as the scratch experiment. '
                 'All layers train. The two-class head is new; validation loss selects the checkpoint. '
                 'No accuracy results have been generated for this package.')
        config = f'''MODEL = {key!r}
# Set this to the EXISTING RDN package root on the lab machine.
DATA_ROOT = os.environ.get('RDN_DATA_ROOT', '')
# Optional: old package's pretrained directory for the verified VMamba weights.
WEIGHTS_DIR = os.environ.get('RDN_PRETRAINED_DIR')
CHECKPOINT = {"'/path/to/original/cnn/best_checkpoint.pt'" if continuation else 'None'}
OVERRIDES = {{}}  # Example: {{'batch_size': 4, 'accumulation_steps': 32, 'seed': 43}}
RUN_NAME = None  # Automatic name; choose a new name for changed settings.
RUN_TRAINING = True
RUN_FINAL_TEST = True
REQUIRE_CUDA = True
CONFIG = get_config(MODEL, OVERRIDES)
print(CONFIG)
'''
        launch = '''# Training starts only in this explicit launch cell.
if RUN_TRAINING or RUN_FINAL_TEST:
    if not DATA_ROOT:
        raise ValueError('Set DATA_ROOT or RDN_DATA_ROOT to the existing full RDN package first.')
    result = run_experiment(MODEL, data_root=DATA_ROOT, overrides=OVERRIDES,
                            run_name=RUN_NAME, checkpoint=CHECKPOINT, weights_dir=WEIGHTS_DIR,
                            run_training=RUN_TRAINING, run_test=RUN_FINAL_TEST,
                            require_cuda=REQUIRE_CUDA)
else:
    print('Training and final test are disabled.')
'''
        cells = [nbf.v4.new_markdown_cell('# ' + header + '\n\n' + intro), code(imports, 'imports'),
                 code(config, 'configuration'), nbf.v4.new_markdown_cell('## Run\n\nReview DATA_ROOT and the configuration before Run All. Completed runs are loaded; interrupted runs resume at the last complete epoch. Changing the configuration requires a new run name. The final test is evaluated only after validation-based checkpoint selection.'),
                 code(launch, 'training_launch'), code("if 'result' in globals():\n    print('Saved to:', result['folder'])\n    print(result['test_metrics'])", 'results')]
        write(folder / (f'{i:02d}_{key}.ipynb' if not continuation else '99_custom_cnn_checkpoint_continuation.ipynb'), cells, key)
    write(folder / '00_environment_and_data_check.ipynb', [nbf.v4.new_markdown_cell('# Fine-tuning preflight\n\nChecks the fixed data and GPU. It does not train a classifier or evaluate the held-out test set.'), code(imports, 'imports'),
          code("DATA_ROOT = os.environ.get('RDN_DATA_ROOT', '')\nCHECK_VMAMBA_KERNEL = False\nREQUIRE_CUDA = True", 'configuration'),
          code("from tools.preflight import preflight\nreport = preflight(DATA_ROOT, require_cuda=REQUIRE_CUDA, check_vmamba=CHECK_VMAMBA_KERNEL)\nprint(report)", 'preflight')])
    print('Generated', len(list(folder.glob('*.ipynb'))), 'clean notebooks')
