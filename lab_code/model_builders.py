"""Load published weights or an explicitly supplied checkpoint; never fall back to random training."""
from functools import partial
from pathlib import Path
import hashlib
import json

import timm
import torch
from torch import nn

from .runtime import FrozenClassifier, sha256_file
from .scratch_models import ScratchCNN, VisionTransformerTiny

VMAMBA_URL = ('https://github.com/MzeroMiko/VMamba/releases/download/'
              '%23v2cls/vssm_base_0229_ckpt_epoch_237.pth')
VMAMBA_SHA256 = '61aec3c5a67cc351dbb8a04c0a3ff76de213e01ff387cc349f94e497ce377452'


def state_digest(model):
    """Record the exact initial tensors, including the seeded two-class head."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        values = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def transfer_tiny(source, target):
    """Map timm ViT-Tiny to the original grayscale MultiheadAttention model.

    RGB projection kernels are summed, so equal-channel RGB inputs and a
    grayscale input have the same patch embedding. No backbone tensors are
    skipped. The new task head is deliberately left at its seeded initialization.
    """
    state = source.state_dict()
    mapped = {
        'patch_embedding.projection.weight': state['patch_embed.proj.weight'].sum(1, keepdim=True),
        'patch_embedding.projection.bias': state['patch_embed.proj.bias'],
        'class_token': state['cls_token'], 'position_embedding': state['pos_embed'],
        'norm.weight': state['norm.weight'], 'norm.bias': state['norm.bias'],
    }
    for i, block in enumerate(target.blocks):
        before, after = f'blocks.{i}.', f'blocks.{i}.'
        for dst, src in [('norm1', 'norm1'), ('norm2', 'norm2'),
                         ('attention.out_proj', 'attn.proj'),
                         ('mlp.layers.0', 'mlp.fc1'), ('mlp.layers.3', 'mlp.fc2')]:
            for suffix in ('weight', 'bias'):
                mapped[after + dst + '.' + suffix] = state[before + src + '.' + suffix]
        mapped[after + 'attention.in_proj_weight'] = state[before + 'attn.qkv.weight']
        mapped[after + 'attention.in_proj_bias'] = state[before + 'attn.qkv.bias']
        block.norm1.eps = source.blocks[i].norm1.eps
        block.norm2.eps = source.blocks[i].norm2.eps
    target.norm.eps = source.norm.eps
    expected = set(target.state_dict()) - {'head.weight', 'head.bias'}
    if set(mapped) != expected:
        raise RuntimeError('ViT-Tiny transfer does not cover exactly every backbone tensor')
    current = target.state_dict()
    current.update(mapped)
    target.load_state_dict(current, strict=True)
    return target


def _vmamba(config, load_weights, weights_dir):
    from . import vmamba_official as official
    if config['require_fast_vmamba'] and not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError('VMamba needs selective_scan_cuda. Run the VMamba preflight; see README.')
    if config['architecture'] != 'vmamba_base_s2l15':
        raise ValueError('This experiment requires the VMamba-Base s2l15 architecture')
    backbone = official.vmamba_base_s2l15(pretrained=False)
    provenance = {'source': VMAMBA_URL, 'expected_checkpoint_sha256': VMAMBA_SHA256}
    if load_weights:
        path = Path(weights_dir) / 'vmamba_base_s2l15_imagenet.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            temporary = path.with_suffix('.downloading')
            torch.hub.download_url_to_file(VMAMBA_URL, str(temporary), hash_prefix=VMAMBA_SHA256)
            temporary.replace(path)
        if sha256_file(path) != VMAMBA_SHA256:
            raise ValueError('VMamba checkpoint SHA-256 mismatch; refusing to load it')
        state = torch.load(path, map_location='cpu', weights_only=True)
        backbone.load_state_dict(state.get('model', state), strict=True)
        provenance['checkpoint_sha256'] = VMAMBA_SHA256
    backbone.classifier.head = nn.Identity()
    backend = 'mamba' if official.WITH_SELECTIVESCAN_MAMBA else 'torch'
    for module in backbone.modules():
        if isinstance(module, official.SS2D):
            module.forward_core = partial(module.forward_corev2, force_fp32=False, no_einsum=True,
                                          selective_scan_backend=backend, scan_force_torch=True)
        if isinstance(module, official.VSSBlock):
            module.use_checkpoint = config['activation_checkpointing']
    return FrozenClassifier(backbone, backbone.num_features, train_backbone=True), provenance


def build_model(config, *, load_weights=True, checkpoint=None, weights_dir='weights'):
    mode, key = config['initialization'], config['model']
    if mode not in ('pretrained', 'checkpoint'):
        raise ValueError('Choose pretrained or checkpoint initialization; random fallback is disabled')
    if not config['train_backbone']:
        raise ValueError('This experiment fine-tunes all layers; train_backbone must be true')
    if key == 'cnn' and mode != 'checkpoint':
        raise ValueError('The custom CNN has no published pretrained weights. Supply its checkpoint '
                         'or choose cnn_finetuned, the EfficientNet-B0 pretrained CNN reference.')
    use_pretrained = mode == 'pretrained' and load_weights
    if key == 'cnn':
        model, provenance = ScratchCNN(classes=2), {'source': 'user-supplied custom CNN checkpoint'}
    elif key == 'vit_tiny':
        model = VisionTransformerTiny(classes=2)
        # Keep the pretrained normalization epsilon in checkpoint-continuation
        # builds too; it is part of this fine-tuning architecture definition.
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = 1e-6 if mode == 'pretrained' else 1e-5
        provenance = {'source': 'timm/' + config['architecture'], 'rgb_to_grayscale': 'sum projection kernels'}
        if use_pretrained:
            source = timm.create_model(config['architecture'], pretrained=True, num_classes=0,
                                       img_size=config['image_size'])
            model = transfer_tiny(source, model)
            del source
    elif key == 'vmamba_base':
        model, provenance = _vmamba(config, use_pretrained, weights_dir)
    else:
        kwargs = {'pretrained': use_pretrained, 'num_classes': 0 if key == 'dinov2' else 2}
        if config['architecture'].startswith('vit_'):
            kwargs.update(img_size=config['image_size'], drop_rate=0.1)
        backbone = timm.create_model(config['architecture'], **kwargs)
        model = FrozenClassifier(backbone, backbone.num_features, train_backbone=True) if key == 'dinov2' else backbone
        provenance = {'source': 'timm/' + config['architecture']}
        if config['activation_checkpointing'] and hasattr(backbone, 'set_grad_checkpointing'):
            backbone.set_grad_checkpointing(True)
    if mode == 'checkpoint' and load_weights:
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError('Checkpoint continuation requires --checkpoint /path/to/best_checkpoint.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        tensors = state.get('model', state.get('state_dict', state))
        model.load_state_dict(tensors, strict=True)
        provenance = {'source': 'user-supplied trained checkpoint', 'checkpoint_sha256': sha256_file(checkpoint),
                      'source_epoch': state.get('epoch')}
    model.requires_grad_(True)
    if load_weights:
        provenance['initial_state_sha256'] = state_digest(model)
    return model, provenance


def classifier_parameters(model):
    if isinstance(model, ScratchCNN):
        return model.classifier.parameters()
    if hasattr(model, 'head') and isinstance(model.head, nn.Module):
        return model.head.parameters()
    return model.get_classifier().parameters()
