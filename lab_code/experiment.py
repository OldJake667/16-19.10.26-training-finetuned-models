"""One entry point for CLI and notebooks. Importing this module does not train or download."""
from pathlib import Path
import copy
import json
import os
import re

import torch

from .model_builders import build_model
from .runtime import (atomic_json, build_training_loaders, config_identity, evaluate_test,
                      fit, load_full_data, seed_everything, select_device, sha256_file)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ('cnn_finetuned', 'vit_tiny', 'resnet50', 'resnet101', 'resnet152',
                  'vit_small', 'vit_base', 'vit_large', 'vmamba_base')


def get_config(model_key, overrides=None):
    configs = json.loads((ROOT / 'MODEL_CONFIGS.json').read_text())
    if model_key not in configs:
        raise ValueError(f'Unknown model: {model_key}; choose from {list(configs)}')
    config = copy.deepcopy(configs[model_key])
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        if overrides.get('model', model_key) != model_key:
            raise ValueError('Cannot change the model key through overrides')
        config.update(overrides)
    if config['initialization'] not in ('pretrained', 'checkpoint'):
        raise ValueError('initialization must be pretrained or checkpoint')
    if not config['train_backbone']:
        raise ValueError('Fine-tuning requires train_backbone=True')
    for key in ('batch_size', 'accumulation_steps', 'max_epochs', 'early_stopping_patience'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= config['warmup_epochs'] < config['max_epochs']:
        raise ValueError('warmup_epochs must be nonnegative and smaller than max_epochs')
    if config['learning_rate'] <= 0 or config['head_learning_rate'] <= 0:
        raise ValueError('Learning rates must be positive')
    return config


def run_experiment(model_key, *, data_root=None, overrides=None, run_name=None,
                   checkpoint=None, weights_dir=None, run_training=True,
                   run_test=True, require_cuda=True):
    config = get_config(model_key, overrides)
    data_root = Path(data_root or os.environ.get('RDN_DATA_ROOT', ROOT)).expanduser().resolve()
    if not (data_root / 'data/full_manifest.csv').is_file():
        raise FileNotFoundError('Set DATA_ROOT / --data-root / RDN_DATA_ROOT to the existing RDN '
                                'package containing data/full_manifest.csv and PACKAGE_MANIFEST.json')
    device = select_device(require_cuda=require_cuda)
    if device.type == 'cpu':
        config['precision'] = 'fp32'
    if model_key == 'vmamba_base' and device.type != 'cuda':
        raise RuntimeError('Full VMamba training requires CUDA and the fast scan kernel')
    name = run_name or (f"seed_{config['seed']}_{config['initialization']}_finetune_all_layers_"
                        f"epochs_{config['max_epochs']}")
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError('run_name must be a simple folder name, without slashes')
    folder = ROOT / 'results' / model_key / name
    folder.mkdir(parents=True, exist_ok=True)
    if not run_training and not (folder / 'TRAINING_COMPLETE.json').is_file():
        raise RuntimeError('Evaluation-only mode requires an already completed run with the same settings')
    partitions, identity = load_full_data(data_root)
    identity['support_code_sha256'] = {p.name: sha256_file(p) for p in (ROOT / 'lab_code').glob('*.py')}
    identity['model_configs_sha256'] = sha256_file(ROOT / 'MODEL_CONFIGS.json')
    seed_everything(config['seed'])
    continuing = (folder / 'last_checkpoint.pt').exists() or (folder / 'TRAINING_COMPLETE.json').exists()
    if continuing:
        if not (folder / 'model_info.json').is_file():
            raise ValueError('Saved run is missing initialization provenance')
        model_info = json.loads((folder / 'model_info.json').read_text())
        provenance = model_info['initialization_provenance']
        if checkpoint is not None and provenance.get('checkpoint_sha256') != sha256_file(checkpoint):
            raise ValueError('The supplied source checkpoint differs from this run\'s initialization')
    else:
        model_info, provenance = None, None
    # Rebuild architectures without any download on resume; fit() then loads
    # the saved model/optimizer/scaler/RNG state strictly before taking a step.
    model, new_provenance = build_model(config, load_weights=not continuing, checkpoint=checkpoint,
                                       weights_dir=weights_dir or ROOT / 'weights')
    provenance = provenance or new_provenance
    identity['initialization'] = provenance
    if model_info is None:
        model_info = {'architecture': config['architecture'], 'model': model_key,
                      'total_parameters': sum(p.numel() for p in model.parameters()),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'training_mode': 'all_layers_finetune', 'pretrained': config['initialization'] == 'pretrained',
                      'initialization_provenance': provenance, 'input_size': config['image_size'],
                      'input_channels': config['channels'], 'normalization': config['normalization'],
                      'interpolation': config['interpolation'], 'physical_batch_size': config['batch_size'],
                      'effective_batch_size': config['batch_size'] * config['accumulation_steps']}
    existing = folder / 'run_config.json'
    if existing.exists() and json.loads(existing.read_text()) != config_identity(config, identity):
        raise ValueError('Configuration, data or source code changed. Choose a different run_name.')
    training_model, loaders, cached = build_training_loaders(model, data_root, partitions, config, device, identity)
    assert not cached, 'Fine-tuning must recompute the backbone; cached features are disabled'
    print(f"{model_key}: {model_info['trainable_parameters']:,} trainable parameters; "
          f"initialization={config['initialization']}; output={folder}")
    if run_training:
        history = fit(training_model, loaders, config, device, folder, identity, model_info)
    else:
        state = torch.load(folder / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
        if state['contract'] != config_identity(config, identity):
            raise ValueError('Saved checkpoint contract differs from requested evaluation')
        training_model.load_state_dict(state['model'], strict=True)
        history = json.loads((folder / 'history.json').read_text())
    metrics = evaluate_test(model, training_model, data_root, partitions, config, device,
                            folder, identity, cached=False) if run_test else None
    return {'folder': folder, 'history': history, 'test_metrics': metrics, 'model_info': model_info}
