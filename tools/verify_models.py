"""Inspect architectures/data; optionally load the pretrained EfficientNet-B0 CNN.

Does not train on RDN or run classifier inference on any dataset images.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import timm
import torch
from lab_code.experiment import DEFAULT_MODELS, get_config
from lab_code.model_builders import build_model
from lab_code.runtime import atomic_json, layout_dataset, load_full_data


def verify(data_root=None, pretrained_cnn=False):
    torch.set_num_threads(2)
    report = {'python': platform.python_version(), 'torch': str(torch.__version__), 'timm': timm.__version__,
              'device': 'CPU', 'gpu_verified': False, 'research_training_executed': False,
              'dataset_model_inference_executed': False, 'architectures': {}}
    original_linspace = torch.linspace
    def cpu_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)
    torch.linspace = cpu_linspace
    try:
        for key in DEFAULT_MODELS:
            c = get_config(key); c['require_fast_vmamba'] = False
            with torch.device('meta'):
                model, _ = build_model(c, load_weights=False)
            total = sum(p.numel() for p in model.parameters())
            trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert total == trained and total > 0
            report['architectures'][key] = {'total_parameters': total, 'trainable_parameters': trained,
                                          'created_on_meta_without_weights': True}
            del model; gc.collect()
    finally:
        torch.linspace = original_linspace
    if data_root:
        partitions, identity = load_full_data(data_root)
        report['verified_split'] = {k: len(v) for k,v in partitions.items()}
        report['verified_data_sha256'] = identity
        for key in DEFAULT_MODELS:
            x,y,_ = layout_dataset(data_root, partitions['train'][:1], get_config(key))[0]
            c = get_config(key)
            assert tuple(x.shape) == (c['channels'],c['image_size'],c['image_size'])
            assert torch.isfinite(x).all() and y in (0,1)
        report['all_preprocessing_shapes_checked'] = True
    if pretrained_cnn:
        torch.manual_seed(42)
        model, provenance = build_model(get_config('cnn_finetuned'))
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2,3,224,224))
        assert output.shape == (2,2) and torch.isfinite(output).all()
        report['pretrained_cnn'] = {'published_weights_loaded': True, 'synthetic_forward_passed': True,
                                    'provenance': provenance}
    report['limitations'] = ['CUDA hardware and selective-scan forward/backward require GPU preflight',
                             'No full-data or subset research training or held-out evaluation performed',
                             'Recipes and physical batches need validation on the target GPU']
    target = ROOT / ('VERIFICATION_PRETRAINED.json' if pretrained_cnn else 'VERIFICATION.json')
    atomic_json(target, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root');p.add_argument('--pretrained-cnn',action='store_true')
    a=p.parse_args();verify(a.data_root,a.pretrained_cnn)
