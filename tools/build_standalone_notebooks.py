"""Build complete notebook copies; generated notebooks never import project helpers.

This is a maintenance tool. Notebook users only need the notebook, installed
libraries, the four raw data files, and published pretrained weights.
"""
import ast
import hashlib
import json
from pathlib import Path
import pprint
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'standalone_notebooks'
CONFIGS = json.loads((ROOT / 'MODEL_CONFIGS.json').read_text())
MANIFEST = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
DATA_HASHES = {key: value['sha256'] for key, value in MANIFEST['files'].items()}
SOURCES = {name: (ROOT / 'lab_code' / name).read_text() for name in
           ('runtime.py', 'scratch_models.py', 'model_builders.py', 'vmamba_official.py')}
LEGACY_DIGESTS = sorted(hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / 'lab_code').glob('*.py'))
LEGACY_CONFIG_DIGEST = hashlib.sha256((ROOT / 'MODEL_CONFIGS.json').read_bytes()).hexdigest()


def definitions(filename, names):
    """Copy readable definitions, keeping comments and their original order."""
    source = SOURCES[filename]
    lines = source.splitlines()
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            found[node.name] = '\n'.join(lines[start:node.end_lineno])
    return '\n\n\n'.join(found[name] for name in names) + '\n'


def runtime(*names):
    return definitions('runtime.py', names)


def code(source, role):
    cell = nbf.v4.new_code_cell(textwrap.dedent(source).strip() + '\n')
    cell.metadata['lab_role'] = role
    return cell


def section(cells, title, explanation, source, role):
    cells.extend([nbf.v4.new_markdown_cell('## ' + title + '\n\n' + explanation), code(source, role)])


IMPORTS = '''from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
'''

UTILITIES = runtime('sha256_file', 'atomic_json', 'atomic_checkpoint', 'environment_info',
                    'select_device', 'seed_everything', 'precision_context')
UTILITIES = UTILITIES.replace('see START_HERE.md. Training has not started.',
                              'use a CUDA-enabled Python environment. Training has not started.')
DATA_SOURCE = runtime('load_full_data')
DATA_SOURCE = DATA_SOURCE.replace('    provenance = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))\n    pinned = json.loads((Path(__file__).resolve().parents[1] / "PACKAGE_MANIFEST.json").read_text())\n', '')
DATA_SOURCE = DATA_SOURCE.replace('        if provenance["files"][name]["sha256"] != pinned["files"][name]["sha256"]:\n            raise ValueError(f"Data identity differs from the fixed scratch baseline: {name}")\n', '')
DATA_SOURCE = DATA_SOURCE.replace('provenance["files"][name]["sha256"]', 'DATA_SHA256[name]')
DATA_SOURCE = ('CLASS_NAMES = ["PeakBelow83", "PeakAtOrAbove83"]\n'
               'EXPECTED_COUNTS = {"train": 97200, "validation": 10800, "test": 3000}\n'
               'DATA_SHA256 = ' + pprint.pformat(DATA_HASHES, sort_dicts=False) + '\n\n' + DATA_SOURCE)
DATASET_SOURCE = runtime('LayoutDataset', 'seed_worker', 'make_loader', 'layout_dataset')
LOADERS = '''def build_training_loaders(model, root, partitions, config, device, identity):
    # Recompute the backbone on every step: this is full fine-tuning.
    datasets = {
        split: layout_dataset(root, partitions[split], config)
        for split in ("train", "validation")
    }
    model.to(device)
    loaders = {
        split: make_loader(
            datasets[split], config["batch_size"], split == "train",
            config["seed"], config["num_workers"], device,
        )
        for split in ("train", "validation")
    }
    return model, loaders, False
'''
OPTIMIZER = runtime('optimizer_and_scheduler').replace('    from .model_builders import classifier_parameters\n', '')
CLASSIFIER_PARAMETERS = '''def classifier_parameters(model):
    if hasattr(model, "head"):
        return model.head.parameters()
    return model.get_classifier().parameters()
'''
CHECKPOINTS = runtime('capture_rng', 'restore_rng', 'config_identity') + '''

def notebook_program_identity(folder, config, data_identity):
    """Bind new runs to notebook code; safely continue unchanged older runs.

    Output cells and editable settings do not enter the program hash. Settings
    and exact data hashes are checked separately in every checkpoint contract.
    """
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    excluded = {"configuration", "training_launch", "results", "preflight"}
    implementation = [
        [cell["metadata"]["lab_role"], "".join(cell["source"])]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and cell["metadata"]["lab_role"] not in excluded
    ]
    digest = hashlib.sha256(json.dumps(implementation).encode()).hexdigest()
    identity = {**data_identity, "notebook_code_sha256": digest}
    previous_path = folder / "run_config.json"
    if not previous_path.exists():
        return identity

    previous = json.loads(previous_path.read_text())
    if previous["config"] != config:
        raise ValueError("Saved settings differ. Choose a new RUN_NAME for changed settings.")
    saved = previous["data"]
    if "notebook_code_sha256" in saved:
        if saved != identity:
            raise ValueError("Notebook code or data changed. Choose a new RUN_NAME.")
        return identity

    # An existing run may predate the self-contained notebooks. Accept its
    # original contract only for the exact implementation verified at release.
    compatibility = notebook["metadata"]["standalone"]
    original_data = {key: value for key, value in saved.items()
                     if key not in ("support_code_sha256", "model_configs_sha256")}
    source_digests = sorted(saved.get("support_code_sha256", {}).values())
    if (digest != compatibility["verified_legacy_code_sha256"]
            or source_digests != compatibility["legacy_source_digests"]
            or saved.get("model_configs_sha256") != compatibility["legacy_config_digest"]
            or original_data != data_identity):
        raise ValueError("Older run is not compatible with this notebook. Choose a new RUN_NAME.")
    record = folder / "notebook_execution.json"
    if record.exists() and json.loads(record.read_text())["notebook_code_sha256"] != digest:
        raise ValueError("This run previously used different notebook code. Choose a new RUN_NAME.")
    atomic_json(record, {"notebook_code_sha256": digest,
                         "notebook": NOTEBOOK_PATH.name,
                         "continuing_original_checkpoint_contract": True})
    print("Using the compatible existing run; notebook execution identity recorded.")
    return saved
'''
TEST_SOURCE = runtime('evaluate_test')
start = TEST_SOURCE.index('    if cached:\n')
end = TEST_SOURCE.index('    loader = make_loader', start)
TEST_SOURCE = TEST_SOURCE[:start] + '    dataset = layout_dataset(root, rows, config)\n' + TEST_SOURCE[end:]

VALIDATE_CONFIG = '''def validate_config(config):
    if config["model"] != MODEL:
        raise ValueError("MODEL and CONFIG['model'] must match.")
    if config["initialization"] != "pretrained" or not config["train_backbone"]:
        raise ValueError("This notebook uses published weights and trains all layers.")
    for name in ("batch_size", "accumulation_steps", "max_epochs", "early_stopping_patience"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 <= config["warmup_epochs"] < config["max_epochs"]:
        raise ValueError("Warmup must be shorter than the epoch budget.")
    if config["learning_rate"] <= 0 or config["head_learning_rate"] <= 0:
        raise ValueError("Learning rates must be positive.")
'''

RUN_EXPERIMENT = '''def run_experiment(config, *, data_root, run_name=None, weights_dir=None,
                   run_training=True, run_test=True, require_cuda=True):
    config = copy.deepcopy(config)
    validate_config(config)
    data_root = Path(data_root).expanduser().resolve()
    device = select_device(require_cuda=require_cuda)
    if device.type == "cpu":
        config["precision"] = "fp32"
    if MODEL == "vmamba_base" and device.type != "cuda":
        raise RuntimeError("Full VMamba training requires CUDA and the fast scan kernel.")
    name = run_name or (
        f"seed_{config['seed']}_{config['initialization']}_finetune_all_layers_"
        f"epochs_{config['max_epochs']}"
    )
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                       for character in name) or name in (".", ".."):
        raise ValueError("RUN_NAME must be a simple folder name without slashes.")
    folder = PROJECT_DIR / "results" / MODEL / name
    folder.mkdir(parents=True, exist_ok=True)
    if not run_training and not (folder / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("Evaluation-only mode requires a completed run with matching settings.")
    partitions, identity = load_full_data(data_root)
    seed_everything(config["seed"])
    continuing = (folder / "last_checkpoint.pt").exists() or (folder / "TRAINING_COMPLETE.json").exists()
    if continuing:
        if not (folder / "model_info.json").is_file():
            raise ValueError("Saved run is missing initialization provenance.")
        model_info = json.loads((folder / "model_info.json").read_text())
        provenance = model_info["initialization_provenance"]
    else:
        model_info, provenance = None, None
    model, new_provenance = build_model(config, load_weights=not continuing,
                                       weights_dir=weights_dir or PROJECT_DIR / "weights")
    provenance = provenance or new_provenance
    identity["initialization"] = provenance
    identity = notebook_program_identity(folder, config, identity)
    if model_info is None:
        model_info = {
            "architecture": config["architecture"], "model": MODEL,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training_mode": "all_layers_finetune", "pretrained": True,
            "initialization_provenance": provenance,
            "input_size": config["image_size"], "input_channels": config["channels"],
            "normalization": config["normalization"], "interpolation": config["interpolation"],
            "physical_batch_size": config["batch_size"],
            "effective_batch_size": config["batch_size"] * config["accumulation_steps"],
        }
    training_model, loaders, cached = build_training_loaders(
        model, data_root, partitions, config, device, identity,
    )
    print(f"{MODEL}: {model_info['trainable_parameters']:,} trainable parameters; saved to {folder}")
    if run_training:
        history = fit(training_model, loaders, config, device, folder, identity, model_info)
    else:
        state = torch.load(folder / "best_checkpoint.pt", map_location="cpu", weights_only=True)
        if state["contract"] != config_identity(config, identity):
            raise ValueError("Saved checkpoint contract differs from requested evaluation.")
        training_model.load_state_dict(state["model"], strict=True)
        history = json.loads((folder / "history.json").read_text())
    metrics = evaluate_test(model, training_model, data_root, partitions, config, device,
                            folder, identity, cached=False) if run_test else None
    return {"folder": folder, "history": history, "test_metrics": metrics, "model_info": model_info}
'''

STANDARD_BUILDER = '''def build_model(config, *, load_weights=True, weights_dir=None):
    # timm supplies the standard architecture and published ImageNet weights.
    # num_classes=2 replaces the original classifier with a new task head.
    options = {"pretrained": load_weights, "num_classes": 2}
    if config["architecture"].startswith("vit_"):
        options.update(img_size=config["image_size"], drop_rate=0.1)
    model = timm.create_model(config["architecture"], **options)
    if config["activation_checkpointing"] and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    model.requires_grad_(True)
    provenance = {"source": "timm/" + config["architecture"]}
    if load_weights:
        provenance["initial_state_sha256"] = state_digest(model)
    return model, provenance
'''
TINY_BUILDER = '''def build_model(config, *, load_weights=True, weights_dir=None):
    model = VisionTransformerTiny(classes=2)
    # These epsilons match the published pretrained backbone.
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = 1e-6
    provenance = {"source": "timm/" + config["architecture"],
                  "rgb_to_grayscale": "sum projection kernels"}
    if load_weights:
        source = timm.create_model(config["architecture"], pretrained=True,
                                   num_classes=0, img_size=config["image_size"])
        transfer_tiny(source, model)
        del source
    model.requires_grad_(True)
    if load_weights:
        provenance["initial_state_sha256"] = state_digest(model)
    return model, provenance
'''

SCAN_IMPORTS = '''from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional
import warnings

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_

try:
    import selective_scan_cuda
    WITH_SELECTIVESCAN_MAMBA = True
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False
'''
CROSS_WRAPPERS = '''def cross_scan_fn(x, in_channel_first=True, out_channel_first=True,
                  one_by_one=False, scans=0, force_torch=True):
    return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


def cross_merge_fn(y, in_channel_first=True, out_channel_first=True,
                   one_by_one=False, scans=0, force_torch=True):
    return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
'''

CUDA_SCAN = '''class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, oflex=True, backend=None):
        if backend not in (None, "mamba"):
            raise ValueError("This notebook uses the Mamba selective-scan extension.")
        ctx.delta_softplus = delta_softplus
        out, state, *_ = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, delta_bias, delta_softplus,
        )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, dbias, *_ = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, state,
            None, None, ctx.delta_softplus, False,
        )
        return du, ddelta, dA, dB, dC, dD, dbias, None, None, None
'''


def selective_scan_source():
    return (definitions('vmamba_official.py', ('selective_scan_torch',))
            + '\n' + CUDA_SCAN + '\n'
            + definitions('vmamba_official.py', ('selective_scan_fn',)))


def vmamba_sections():
    core = definitions('vmamba_official.py', ('SS2Dv2', 'SS2D', 'VSSBlock', 'VSSM'))
    core = core.replace('class SS2D(nn.Module, SS2Dv0, SS2Dv2):',
                        'class SS2D(nn.Module, SS2Dv2):')
    start = core.index('        if forward_type in ["v0", "v0seq"]:')
    end = core.index('\n    def _load_from_state_dict', start)
    core = core[:start] + '        self.__initv2__(**kwargs)\n' + core[end:]
    # Remove the unrelated FLOP profiling routine (and its fvcore dependency).
    tree = ast.parse(core)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'VSSM':
            lines = core.splitlines()
            method = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == 'flops')
            del lines[method.lineno-1:method.end_lineno]
            core = '\n'.join(lines) + '\n'
            break
    constructor = definitions('vmamba_official.py', ('vmamba_base_s2l15',))
    constructor = constructor.replace('@register_model\n', '')
    constructor = constructor[:constructor.index('    if pretrained:\n')] + '    return model\n'
    wrapper = '''class FullFineTuneClassifier(nn.Module):
    """VMamba backbone and a new two-class head; all layers train."""
    def __init__(self, backbone, feature_dim, train_backbone=True):
        super().__init__()
        if not train_backbone:
            raise ValueError("This notebook trains the complete backbone.")
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)
        self.requires_grad_(True)

    def forward(self, images):
        return self.head(self.backbone(images))
'''
    builder = definitions('model_builders.py', ('_vmamba',))
    builder = builder.replace('    from . import vmamba_official as official\n', '').replace('official.', '')
    builder = builder.replace('FrozenClassifier', 'FullFineTuneClassifier')
    builder = builder.replace('Run the VMamba preflight; see README.', 'Run the kernel check in the environment notebook.')
    builder += '''\n\ndef build_model(config, *, load_weights=True, weights_dir=None):
    model, provenance = _vmamba(config, load_weights, Path(weights_dir or WEIGHTS_DIR))
    model.requires_grad_(True)
    if load_weights:
        provenance["initial_state_sha256"] = state_digest(model)
    return model, provenance
'''
    constants = ("VMAMBA_URL = 'https://github.com/MzeroMiko/VMamba/releases/download/"
                 "%23v2cls/vssm_base_0229_ckpt_epoch_237.pth'\n"
                 "VMAMBA_SHA256 = '61aec3c5a67cc351dbb8a04c0a3ff76de213e01ff387cc349f94e497ce377452'\n\n")
    return [
        ('4a. VMamba scan operations', 'PyTorch cross-scan/merge and their backward passes. The original MIT license is included below.',
         SCAN_IMPORTS + '\n' + definitions('vmamba_official.py', ('cross_scan_fwd', 'cross_merge_fwd', 'cross_scan1b1_fwd', 'cross_merge1b1_fwd', 'CrossScanF', 'CrossMergeF')) + CROSS_WRAPPERS, 'vmamba_cross_scan'),
        ('4b. Selective scan', 'The compiled Mamba CUDA extension accelerates selective scan. The tensor implementation is retained for kernel checks, not full CPU training.',
         selective_scan_source(), 'vmamba_selective_scan'),
        ('4c. VMamba building blocks', 'Channel-first layers and state-space initialization used by the official Base checkpoint.',
         definitions('vmamba_official.py', ('Linear', 'LayerNorm', 'PatchMerge', 'Permute', 'SoftmaxSpatial', 'Mlp', 'mamba_init')), 'vmamba_layers'),
        ('4d. VMamba backbone', 'The state-space blocks and hierarchical backbone. Parameter names are kept compatible with the published weights and your saved runs.',
         core, 'vmamba_backbone'),
        ('4e. Load the Base weights', 'Load and verify the complete 1,000-class backbone, then attach the new two-class head. Every layer remains trainable.',
         constructor + '\n' + wrapper + '\n' + constants + definitions('model_builders.py', ('state_digest',)) + builder, 'model'),
    ]


def configuration(key, filename):
    settings = f'''MODEL = {key!r}
NOTEBOOK_FILENAME = {filename!r}
# Change these paths if your notebook or dataset is stored elsewhere.
PROJECT_DIR = Path.cwd().parent if Path.cwd().name in ("notebooks", "standalone_notebooks") else Path.cwd()
DATA_ROOT = Path(os.environ.get("RDN_DATA_ROOT", Path.home() / "Desktop" / "training_models"))
WEIGHTS_DIR = Path(os.environ.get("RDN_PRETRAINED_DIR", DATA_ROOT / "pretrained"))
NOTEBOOK_PATH = next((p for p in (
    PROJECT_DIR / "standalone_notebooks" / NOTEBOOK_FILENAME,
    Path.cwd() / NOTEBOOK_FILENAME,
    PROJECT_DIR / NOTEBOOK_FILENAME,
) if p.is_file()), PROJECT_DIR / "standalone_notebooks" / NOTEBOOK_FILENAME)

# None uses the same automatic run name as the original notebook.
# Give a new name when changing the recipe; existing runs are checked strictly.
RUN_NAME = None
RUN_TRAINING = True
RUN_FINAL_TEST = True
REQUIRE_CUDA = True

# Edit the values here. No external configuration file is loaded.
CONFIG = {pprint.pformat(CONFIGS[key], sort_dicts=False, width=92)}

print(f"{{MODEL}}: maximum {{CONFIG['max_epochs']}} epochs, patience {{CONFIG['early_stopping_patience']}}")
print("Effective batch:", CONFIG["batch_size"] * CONFIG["accumulation_steps"])
'''
    return settings


def implementation_digest(notebook):
    excluded = {'configuration', 'training_launch', 'results', 'preflight'}
    implementation = [[cell.metadata['lab_role'], cell.source]
                      for cell in notebook.cells if cell.cell_type == 'code'
                      and cell.metadata['lab_role'] not in excluded]
    return hashlib.sha256(json.dumps(implementation).encode()).hexdigest()


def write_notebook(filename, cells, key):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={'display_name': 'RDN fine-tuning', 'language': 'python', 'name': 'rdn-finetune'},
        language_info={'name': 'python'},
        experiment={'model': key, 'training_executed': False},
        standalone={'version': 1, 'project_helper_imports': False,
                    'legacy_source_digests': LEGACY_DIGESTS,
                    'legacy_config_digest': LEGACY_CONFIG_DIGEST},
    )
    notebook.metadata['standalone']['verified_legacy_code_sha256'] = implementation_digest(notebook)
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT / filename)


def build_model_notebook(index, key):
    name = key if key.endswith('_finetuned') else key + '_finetuned'
    filename = f'{index:02d}_{name}.ipynb'
    labels = {'cnn_finetuned': 'CNN fine-tuned (EfficientNet-B0)',
              'vit_tiny': 'ViT-Tiny fine-tuned', 'vit_small': 'ViT-Small fine-tuned',
              'vit_base': 'ViT-Base fine-tuned', 'vit_large': 'ViT-Large fine-tuned',
              'resnet50': 'ResNet50 fine-tuned', 'resnet101': 'ResNet101 fine-tuned',
              'resnet152': 'ResNet152 fine-tuned', 'vmamba_base': 'VMamba-Base fine-tuned'}
    title = labels[key]
    cells = [nbf.v4.new_markdown_cell(
        f'# {title}: ImageNet weights → RDN full fine-tuning\n\n'
        'This notebook contains its configuration, data checks, model construction, training loop, checkpoint resume, plots, and final test evaluation. '
        'It imports installed libraries only; no project helper scripts or external configuration files are needed. '
        'The four raw dataset files and published weights remain separate.\n\n'
        '**Run one GPU notebook at a time.** To switch from an older notebook, finish or interrupt its current epoch, shut down its kernel, '
        'then open this notebook and use **Restart Kernel and Run All Cells**. Completed epochs can resume from the compatible saved run.\n\n'
        '**This is a code-organization change.** Architecture, initialization, split, preprocessing, learning rates, and stopping settings are preserved. '
        'The earlier ViT convergence problems have not been silently changed. Save notebook edits before starting so code identity can be recorded.')]
    section(cells, '1. Imports', 'Installed packages: PyTorch, torchvision, timm, NumPy, scikit-learn, matplotlib, Pillow, and tqdm.', IMPORTS, 'imports')
    section(cells, '2. Settings', 'Review paths, epoch limit, patience, batch size, and learning rates here. The backbone and new head both train.', configuration(key, filename), 'configuration')
    section(cells, '3a. Runtime utilities', 'Device selection, precision, reproducibility, and atomic saving. Defining these functions does not start training.', UTILITIES, 'utilities')
    section(cells, '3b. Check the dataset', 'Pin the original four file hashes, binary target, and fixed 97,200 / 10,800 / 3,000 split. Both validation classes are retained.', DATA_SOURCE, 'data_integrity')
    section(cells, '3c. Images and loaders', 'Resize the complete geometry, apply the recorded normalization, and build deterministic loaders. Only the training loader shuffles.', DATASET_SOURCE + '\n' + LOADERS, 'datasets')
    if key == 'vmamba_base':
        license_text = next((ROOT / 'licenses').glob('*')).read_text()
        cells.append(nbf.v4.new_markdown_cell('### VMamba attribution\n\nAdapted from [the official VMamba implementation](https://github.com/MzeroMiko/VMamba). Unused profiling, benchmarks, and Triton cross-scan kernels are omitted; this experiment uses the PyTorch cross-scan path.\n\n```text\n' + license_text.rstrip() + '\n```'))
        for args in vmamba_sections():
            section(cells, *args)
    elif key == 'vit_tiny':
        section(cells, '4a. ViT-Tiny architecture', 'The original grayscale ViT-Tiny: 16×16 patches, width 192, 12 transformer blocks, three attention heads, and two output classes.', definitions('scratch_models.py', ('PatchEmbedding', 'MLP', 'TransformerBlock', 'VisionTransformerTiny')), 'tiny_architecture')
        section(cells, '4b. ImageNet weight transfer', 'Transfer every backbone tensor. Sum RGB patch kernels for equivalent grayscale input and initialize a fresh two-class head. Missing weights cause an error.', definitions('model_builders.py', ('state_digest', 'transfer_tiny')) + '\n' + TINY_BUILDER, 'model')
    else:
        section(cells, '4. Model and ImageNet weights', f"Architecture: `{CONFIGS[key]['architecture']}`. timm constructs the standard backbone; the two-class classifier is new. All parameters train. Resume loads your saved weights instead of downloading again.", definitions('model_builders.py', ('state_digest',)) + '\n' + STANDARD_BUILDER, 'model')
    section(cells, '5. Optimizer and schedule', 'AdamW uses separate backbone/head learning rates, excludes bias/norm/token embeddings from decay, warms up for two epochs, and then applies cosine decay.', CLASSIFIER_PARAMETERS + '\n' + OPTIMIZER, 'optimizer')
    section(cells, '6a. One training epoch', 'Accumulate sample-weighted gradients to the effective batch size. Clip gradients and stop on non-finite loss.', runtime('train_one_epoch'), 'train_epoch')
    section(cells, '6b. Validation metrics', 'Evaluation mode disables dropout. Log loss, accuracy, balanced accuracy, F1, AUC, confusion matrix, and prediction counts; one-class predictions remain visible.', runtime('evaluate'), 'evaluate')
    section(cells, '7a. Save results and plot learning curves', 'Training/validation accuracy and loss are plotted per epoch. The held-out test is evaluated after selecting the checkpoint by validation loss.', runtime('save_predictions', 'save_history', 'plot_history'), 'history')
    section(cells, '7b. Checkpoints and resume identity', 'Save optimizer, scheduler, scaler, RNG, and loader state. Changed settings or implementation require a new run name. Compatible existing runs retain their original contract and record the notebook execution separately.', CHECKPOINTS, 'checkpoints')
    section(cells, '8. Training loop and early stopping', 'Select the lowest-validation-loss checkpoint. Stop at the maximum epoch count or after the configured patience. An interrupted partial epoch repeats on resume.', runtime('fit'), 'fit')
    section(cells, '9. Final held-out test', 'Evaluate the validation-selected checkpoint once and reuse its recorded test result. Save predictions, metrics, and the test confusion matrix.', TEST_SOURCE, 'test_evaluation')
    section(cells, '10. Assemble the experiment', 'Connect the definitions above. No training occurs until the next cell.', VALIDATE_CONFIG + '\n' + RUN_EXPERIMENT, 'experiment')
    section(cells, '11. Run training and final evaluation', 'This cell starts the requested work. Keep the kernel alive while it runs. Set RUN_FINAL_TEST=False while tuning a recipe.', '''if RUN_TRAINING or RUN_FINAL_TEST:
    result = run_experiment(
        CONFIG, data_root=DATA_ROOT, weights_dir=WEIGHTS_DIR, run_name=RUN_NAME,
        run_training=RUN_TRAINING, run_test=RUN_FINAL_TEST, require_cuda=REQUIRE_CUDA,
    )
else:
    print("Training and final test are disabled.")
''', 'training_launch')
    section(cells, '12. View results', 'Display the saved learning curves and final test summary. A test score is a final measurement rather than a per-epoch curve.', '''if "result" in globals():
    print("Results:", result["folder"])
    plot_history(result["folder"], show=True)
    if result["test_metrics"] is not None:
        print(json.dumps(result["test_metrics"], indent=2))
        display(Image.open(result["folder"] / "test_confusion_matrix.png"))
''', 'results')
    write_notebook(filename, cells, key)


def build_preflight():
    cells = [nbf.v4.new_markdown_cell('# Environment and RDN data check\n\nSelf-contained device, dataset integrity, class-count, and optional VMamba kernel checks. No classifier is trained and no dataset images are scored.')]
    section(cells, '1. Imports', 'Use the same installed environment as the model notebooks.', IMPORTS, 'imports')
    section(cells, '2. Settings', 'The launcher sets RDN_DATA_ROOT. Change DATA_ROOT here when running independently.', '''DATA_ROOT = Path(os.environ.get("RDN_DATA_ROOT", Path.home() / "Desktop" / "training_models"))
REQUIRE_CUDA = True
CHECK_VMAMBA_KERNEL = False
''', 'configuration')
    section(cells, '3. Device and data checks', 'Verify the exact raw data and label definition without training.', UTILITIES + '\n' + DATA_SOURCE, 'data_integrity')
    preflight_source = (ROOT / 'tools/preflight.py').read_text()
    node = next(n for n in ast.parse(preflight_source).body if isinstance(n, ast.FunctionDef) and n.name == 'check_scan')
    check = '\n'.join(preflight_source.splitlines()[node.lineno-1:node.end_lineno])
    check = check.replace('    from lab_code import vmamba_official as official\n', '').replace('official.', '')
    section(cells, '4. Optional VMamba kernel definitions', 'Only the optional kernel test uses these definitions. Set CHECK_VMAMBA_KERNEL=True before running the final cell to test CUDA forward/backward.', SCAN_IMPORTS + '\n' + selective_scan_source() + '\n' + check, 'vmamba_kernel')
    section(cells, '5. Run checks', 'Report both class counts in every split. These are integrity checks, not model accuracy.', '''device = select_device(require_cuda=REQUIRE_CUDA)
partitions, identity = load_full_data(DATA_ROOT)
if CHECK_VMAMBA_KERNEL:
    check_scan()
report = {
    "environment": environment_info(device),
    "split": {name: len(rows) for name, rows in partitions.items()},
    "class_counts": {name: np.bincount([int(row["label"]) for row in rows], minlength=2).tolist()
                     for name, rows in partitions.items()},
    "vmamba_kernel_checked": CHECK_VMAMBA_KERNEL,
    "training_executed": False,
}
print(json.dumps(report, indent=2))
''', 'preflight')
    write_notebook('00_environment_and_data_check.ipynb', cells, None)


def main():
    OUTPUT.mkdir(exist_ok=True)
    build_preflight()
    for index, key in enumerate(CONFIGS, 1):
        build_model_notebook(index, key)
    print('Generated', 1 + len(CONFIGS), 'self-contained notebooks in', OUTPUT)


if __name__ == '__main__':
    main()
