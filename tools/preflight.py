"""Check RDN integrity and CUDA/VMamba support; never train or score dataset images."""
import argparse
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_code.runtime import load_full_data, select_device, environment_info


def check_scan():
    from lab_code import vmamba_official as official
    if not torch.cuda.is_available() or not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError('Install the supported CUDA selective_scan_cuda extension first; see README')
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        values = [torch.randn((1, 8, 16), device='cuda', dtype=dtype) * .1,
                  torch.randn((1, 8, 16), device='cuda', dtype=dtype) * .1,
                  -torch.rand((8, 1), device='cuda', dtype=torch.float32),
                  torch.randn((1, 4, 1, 16), device='cuda', dtype=dtype) * .1,
                  torch.randn((1, 4, 1, 16), device='cuda', dtype=dtype) * .1,
                  torch.randn(8, device='cuda', dtype=torch.float32) * .1,
                  torch.randn(8, device='cuda', dtype=torch.float32) * .1]
        ref_values = [x.detach().clone().requires_grad_() for x in values]
        actual_values = [x.detach().clone().requires_grad_() for x in values]
        reference = official.selective_scan_fn(*ref_values, delta_softplus=True, oflex=True, backend='torch')
        actual = official.selective_scan_fn(*actual_values, delta_softplus=True, oflex=True, backend='mamba')
        tol = .02 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(actual.float(), reference.float(), rtol=tol, atol=tol)
        reference.float().square().mean().backward(); actual.float().square().mean().backward()
        for ref, out in zip(ref_values, actual_values):
            assert out.grad is not None and torch.isfinite(out.grad).all()
            torch.testing.assert_close(out.grad.float(), ref.grad.float(), rtol=max(tol, .02), atol=max(tol, 1e-4))


def preflight(data_root, require_cuda=True, check_vmamba=False):
    if not data_root:
        raise ValueError('Set DATA_ROOT / --data-root to the existing full RDN package')
    device = select_device(require_cuda=require_cuda)
    partitions, _ = load_full_data(data_root)
    if check_vmamba:
        check_scan()
    return {'environment': environment_info(device), 'split': {k: len(v) for k,v in partitions.items()},
            'vmamba_kernel_checked': check_vmamba, 'training_executed': False,
            'dataset_model_inference_executed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True);p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--vmamba-kernel', action='store_true');a=p.parse_args()
    print(preflight(a.data_root, not a.allow_cpu, a.vmamba_kernel))
