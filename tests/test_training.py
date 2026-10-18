"""Synthetic checks only: no research training or held-out evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import Dataset

from lab_code import runtime as rt
from lab_code.experiment import ROOT, DEFAULT_MODELS, get_config
from lab_code.model_builders import build_model, transfer_tiny


class Samples(Dataset):
    def __init__(self, n=9):
        self.x = torch.randn(n, 4, generator=torch.Generator().manual_seed(19))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], i % 2, f'synthetic-{i}'


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Dropout(.2))
        self.head = nn.Linear(6, 2)
    def forward(self, x): return self.head(self.backbone(x))


def setup():
    rt.seed_everything(42)
    model = Toy()
    device = torch.device('cpu')
    config = get_config('cnn_finetuned', {'max_epochs': 2, 'warmup_epochs': 0, 'precision': 'fp32',
                        'batch_size': 3, 'accumulation_steps': 2, 'learning_rate': .001,
                        'head_learning_rate': .01, 'early_stopping_patience': 5})
    loaders = {s: rt.make_loader(Samples(), 3, s == 'train', 42, 0, device) for s in ('train', 'validation')}
    return model, loaders, config, device


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_tiny_transfer_matches_replicated_rgb(self):
        torch.manual_seed(42)
        config = get_config('vit_tiny')
        target, _ = build_model(config, load_weights=False)
        source = timm.create_model(config['architecture'], pretrained=False, num_classes=2)
        transfer_tiny(source, target)
        source.head.load_state_dict(target.head.state_dict())
        source.eval(); target.eval(); gray = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(source(gray.expand(-1, 3, -1, -1)), target(gray), rtol=2e-4, atol=2e-5)

    def test_checkpoint_continuation_is_strict(self):
        c = json.loads((ROOT/'archived_notebooks/MODEL_CONFIGS.json').read_text())['cnn']
        with self.assertRaises(FileNotFoundError): build_model(c)
        with self.assertRaises(ValueError): build_model({**c, 'initialization': 'pretrained'})
        model, _ = build_model(c, load_weights=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.pt'
            torch.save({'model': model.state_dict(), 'epoch': 10}, path)
            rebuilt, provenance = build_model(c, checkpoint=path)
            self.assertEqual(provenance['source_epoch'], 10)
            self.assertTrue(all(torch.equal(v, rebuilt.state_dict()[k]) for k,v in model.state_dict().items()))
            bad = model.state_dict(); bad['classifier.4.weight'] = torch.randn(3, 128)
            torch.save({'model': bad}, path)
            with self.assertRaises(RuntimeError): build_model(c, checkpoint=path)

    def test_optimizer_groups_cover_every_parameter_once(self):
        model, _, c, _ = setup()
        optimizer, _ = rt.optimizer_and_scheduler(model, c)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        for group in optimizer.param_groups:
            self.assertEqual(group['lr'], .01 if group['name'].startswith('head') else .001)

    def test_accumulation_weights_partial_window_by_samples(self):
        torch.manual_seed(4)
        model = nn.Linear(4, 2); reference = copy.deepcopy(model)
        c = get_config('cnn_finetuned', {'precision': 'fp32', 'batch_size': 3, 'accumulation_steps': 2, 'gradient_clip_norm': 0.0})
        data = Samples(7); loader = rt.make_loader(data, 3, False, 42, 0, torch.device('cpu'))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        rt.train_one_epoch(model, loader, optimizer, torch.amp.GradScaler('cuda', enabled=False), c, torch.device('cpu'))
        opt = torch.optim.SGD(reference.parameters(), lr=.1)
        for indices in (list(range(6)), [6]):
            opt.zero_grad(); x = torch.stack([data[i][0] for i in indices]); y = torch.tensor([data[i][1] for i in indices])
            nn.functional.cross_entropy(reference(x), y).backward(); opt.step()
        for a,b in zip(model.parameters(), reference.parameters()): torch.testing.assert_close(a,b)

    def test_single_class_diagnostics(self):
        class One(nn.Module):
            def forward(self, x): return torch.tensor([0., 1.]).expand(len(x), 2)
        loader = rt.make_loader(Samples(8), 3, False, 42, 0, torch.device('cpu'))
        metrics, _ = rt.evaluate(One(), loader, torch.device('cpu'), get_config('cnn_finetuned', {'precision': 'fp32'}))
        self.assertEqual(metrics['prediction_counts'], [0, 8])
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertTrue(metrics['single_class_predictions'])

    def test_resume_reproduces_uninterrupted_weights(self):
        identity = {'dataset': 'synthetic-unit-test-only'}
        with tempfile.TemporaryDirectory() as tmp:
            whole, resumed = Path(tmp)/'whole', Path(tmp)/'resumed'
            model, loaders, c, device = setup()
            rt.fit(model, loaders, c, device, whole, identity, {'synthetic': True})
            expected = torch.load(whole/'last_checkpoint.pt', weights_only=True)
            model, loaders, c, device = setup(); original = rt.train_one_epoch; calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('simulated interruption')
                return original(*args, **kwargs)
            with patch.object(rt, 'train_one_epoch', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    rt.fit(model, loaders, c, device, resumed, identity, {'synthetic': True})
            self.assertFalse((resumed/'TRAINING_COMPLETE.json').exists())
            model, loaders, c, device = setup()
            rt.fit(model, loaders, c, device, resumed, identity, {'synthetic': True})
            actual = torch.load(resumed/'last_checkpoint.pt', weights_only=True)
            self.assertEqual(actual['epoch'], 2)
            for k,v in expected['model'].items(): self.assertTrue(torch.equal(v,actual['model'][k]), k)
            summary = json.loads((resumed/'training_validation_metrics.json').read_text())
            self.assertFalse(summary['test_evaluated'])
            with self.assertRaises(ValueError):
                rt.fit(model, loaders, {**c,'learning_rate':.004}, device, resumed, identity, {'synthetic':True})

    def test_all_notebooks_are_clean_and_compile(self):
        notebooks = list((ROOT/'notebooks').glob('*.ipynb'))
        self.assertEqual(len(notebooks), 10)
        for path in notebooks:
            n = nbformat.read(path, as_version=4); nbformat.validate(n)
            for cell in n.cells:
                if cell.cell_type == 'code':
                    self.assertIsNone(cell.execution_count); self.assertEqual(cell.outputs, [])
                    compile(cell.source, str(path), 'exec')
        for path in ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts: compile(path.read_text(), str(path), 'exec')

    def test_default_models_have_published_weights(self):
        for key in DEFAULT_MODELS:
            c = get_config(key); self.assertEqual(c['batch_size']*c['accumulation_steps'], 128)
            self.assertTrue(c['train_backbone']); self.assertEqual(c['initialization'], 'pretrained')
            if key != 'vmamba_base': self.assertIsNotNone(timm.get_pretrained_cfg(c['architecture']))

    def test_requested_model_set_and_vmamba_base(self):
        self.assertEqual(set(DEFAULT_MODELS), {'cnn_finetuned', 'vit_tiny', 'resnet50', 'resnet101',
                                             'resnet152', 'vit_small', 'vit_base', 'vit_large', 'vmamba_base'})
        self.assertEqual(set(json.loads((ROOT/'MODEL_CONFIGS.json').read_text())), set(DEFAULT_MODELS))
        config = get_config('vmamba_base', {'require_fast_vmamba': False})
        original_linspace = torch.linspace
        def cpu_linspace(*args, **kwargs):
            return original_linspace(*args, **{**kwargs, 'device': 'cpu'})
        with patch('torch.linspace', cpu_linspace), torch.device('meta'):
            model, _ = build_model(config, load_weights=False)
        self.assertEqual(config['architecture'], 'vmamba_base_s2l15')
        self.assertEqual(model.head.in_features, 1024)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 87534850)


if __name__ == '__main__': unittest.main()
