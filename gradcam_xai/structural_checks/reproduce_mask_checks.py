"""Reproduce the two local classifier checks; never train or relabel images.

python reproduce_mask_checks.py --package-root /path/to/original_bandwidth_xai
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', required=True,
                        help='Original bandwidth XAI folder with checkpoints/ and xai_code/')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    root = Path(args.package_root).expanduser().resolve()
    here = Path(__file__).resolve().parent
    expected = json.loads((here / 'structural_mask_checks.json').read_text())
    checkpoint = root / 'checkpoints/dinov2_best_accuracy.pth'
    if sha256(checkpoint) != expected['checkpoint_sha256']:
        raise ValueError('Use the original archived DINOv2 checkpoint')
    sys.path.insert(0, str(root))
    from xai_code.models import build_model, Predictor
    torch.set_num_threads(4)
    model = build_model(root, 'dinov2', args.device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    predict = Predictor(model, 'dinov2', args.device, batch_size=1)
    results = []
    for case in expected['cases']:
        original = here / case['original_image']
        if sha256(original) != case['original_image_sha256']:
            raise ValueError('Original image changed')
        rgb = np.asarray(Image.open(original).convert('RGB'))
        with np.load(here / case['mask_arrays'], allow_pickle=False) as data:
            target, control = data['target'], data['control']
        if target.shape != rgb.shape[:2] or control.shape != target.shape:
            raise ValueError('Mask shape differs from image')
        if int(target.sum()) != int(control.sum()) or np.any(rgb[target] != 0) or np.any(rgb[control] != 0):
            raise ValueError('Controls must whiten equal numbers of pure black pixels')
        changed = rgb.copy()
        changed[target] = 255
        compared = rgb.copy()
        compared[control] = 255
        if changed.mean() != compared.mean():
            raise ValueError('Native-image mean brightness differs')
        scores = predict([rgb, changed, compared])[:, 1]
        keys = ['baseline_probability_high',
                'mask_upper_right_pronged_branch_probability_high',
                'mask_lower_left_branch_control_probability_high']
        if not np.allclose(scores, [case[k] for k in keys], atol=5e-4, rtol=0):
            raise RuntimeError(f"Prediction drift in {case['sample']}: {scores.tolist()}")
        results.append(dict(sample=case['sample'], **dict(zip(keys, map(float, scores)))))
    if any(not torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('Model state changed')
    print(json.dumps({'cases': results, 'weights_unchanged': True,
                      'physical_bandwidth_simulated': False}, indent=2))


if __name__ == '__main__':
    main()
