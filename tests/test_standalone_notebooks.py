"""Self-contained notebook checks: no research training, dataset scoring, or CUDA use."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import torch
from torch import nn

from lab_code import runtime as original_runtime
from lab_code.experiment import ROOT, DEFAULT_MODELS, get_config
from lab_code.model_builders import build_model as original_build_model
from lab_code import vmamba_official as original_vmamba
from tests.test_training import Samples, setup

FOLDER = ROOT / 'standalone_notebooks'
SKIP_ROLES = {'training_launch', 'results', 'preflight'}


def model_notebook(key):
    suffix = key if key.endswith("_finetuned") else key + "_finetuned"
    return next(FOLDER.glob(f"*_{suffix}.ipynb"))


def load_definitions(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {'__name__': '__main__', 'display': lambda *args: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook.cells:
            if cell.cell_type == 'code' and cell.metadata['lab_role'] not in SKIP_ROLES:
                exec(compile(cell.source, str(path) + ':' + cell.metadata['lab_role'], 'exec'), namespace)
    return namespace


def program_hash(notebook):
    excluded = {'configuration', *SKIP_ROLES}
    program = [[cell.metadata['lab_role'], cell.source] for cell in notebook.cells
               if cell.cell_type == 'code' and cell.metadata['lab_role'] not in excluded]
    return hashlib.sha256(json.dumps(program).encode()).hexdigest()


class StandaloneNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.namespaces = {key: load_definitions(model_notebook(key))
                          for key in DEFAULT_MODELS}

    def test_complete_and_without_project_imports(self):
        paths = sorted([FOLDER/'00_environment_and_data_check.ipynb',
                        *FOLDER.glob('*_finetuned.ipynb')])
        self.assertEqual(len(paths), 10)
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual(program_hash(notebook), notebook.metadata['standalone']['verified_legacy_code_sha256'])
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                tree = ast.parse(cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.level, 0)
                        self.assertNotIn((node.module or '').split('.')[0], ('lab_code', 'tools'))
                    if isinstance(node, ast.Import):
                        self.assertTrue(all(n.name.split('.')[0] not in ('lab_code','tools') for n in node.names))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, ('exec', 'eval', '__import__'))
                self.assertNotIn('__file__', cell.source)
                self.assertNotIn('sys.path', cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertFalse(node.value.endswith('.py'), node.value)
        for key, namespace in self.namespaces.items():
            self.assertEqual(namespace['CONFIG'], get_config(key))
            self.assertEqual(namespace['DATA_SHA256'], {
                name: value['sha256'] for name,value in json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())['files'].items()})

    def test_architecture_parameter_names_and_shapes_match(self):
        original_linspace = torch.linspace
        def cpu_linspace(*args, **kwargs):
            return original_linspace(*args, **{**kwargs, 'device': 'cpu'})
        with patch('torch.linspace', cpu_linspace):
            for key, namespace in self.namespaces.items():
                config = get_config(key)
                config['require_fast_vmamba'] = False
                with torch.device('meta'):
                    new, _ = namespace['build_model'](config, load_weights=False)
                    old, _ = original_build_model(config, load_weights=False)
                new_shapes = {k:tuple(v.shape) for k,v in new.state_dict().items()}
                old_shapes = {k:tuple(v.shape) for k,v in old.state_dict().items()}
                self.assertEqual(new_shapes, old_shapes, key)
                self.assertTrue(all(p.requires_grad for p in new.parameters()),key)

    def test_cached_pretrained_initializations_match_exactly(self):
        with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
            for key in ('cnn_finetuned', 'vit_tiny'):
                config = get_config(key)
                original_runtime.seed_everything(42)
                original, provenance = original_build_model(config)
                original_runtime.seed_everything(42)
                standalone, actual = self.namespaces[key]['build_model'](config)
                self.assertEqual(actual, provenance)
                self.assertTrue(all(torch.equal(v, standalone.state_dict()[k])
                                    for k,v in original.state_dict().items()))

    def test_official_vmamba_base_weights_load_strictly(self):
        weights=Path('/home/murat/Desktop/training_models/pretrained/vmamba_base_s2l15_imagenet.pth')
        if not weights.exists():
            self.skipTest('Official VMamba-Base weights are not cached on this computer')
        namespace=self.namespaces['vmamba_base']
        self.assertEqual(namespace['sha256_file'](weights),namespace['VMAMBA_SHA256'])
        original_linspace=torch.linspace
        def cpu_linspace(*args,**kwargs):
            return original_linspace(*args,**{**kwargs,'device':'cpu'})
        with patch('torch.linspace',cpu_linspace),torch.device('meta'):
            backbone=namespace['vmamba_base_s2l15'](pretrained=False)
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        result=backbone.load_state_dict(checkpoint['model'],strict=True,assign=True)
        self.assertEqual(result.missing_keys,[])
        self.assertEqual(result.unexpected_keys,[])

    def test_preflight_checks_real_data_without_model_inference(self):
        namespace=load_definitions(FOLDER/'00_environment_and_data_check.ipynb')
        partitions,identity=namespace['load_full_data']('/home/murat/Desktop/training_models')
        self.assertEqual({k:len(v) for k,v in partitions.items()},
                         {'train':97200,'validation':10800,'test':3000})
        self.assertEqual(sum(int(row['label']) for row in partitions['validation']),5536)
        self.assertEqual(identity['target'],'compact_response_argmax_ge_83_first_maximum')

    def test_vmamba_reduced_forward_and_backward_match(self):
        namespace = self.namespaces['vmamba_base']
        options = dict(depths=[1,1],dims=8,drop_path_rate=0.,patch_size=4,in_chans=3,
                       num_classes=2,ssm_d_state=1,ssm_ratio=2.,ssm_conv=3,ssm_conv_bias=False,
                       forward_type='v05_noz',norm_layer='ln2d',downsample_version='v3',
                       patchembed_version='v2')
        torch.manual_seed(42); original = original_vmamba.VSSM(**options)
        torch.manual_seed(42); standalone = namespace['VSSM'](**options)
        standalone.load_state_dict(original.state_dict(), strict=True)
        for model, module in ((original,original_vmamba),(standalone,type('Inline',(),namespace))):
            for block in model.modules():
                if isinstance(block,module.SS2D):
                    from functools import partial
                    block.forward_core = partial(block.forward_corev2,force_fp32=False,
                                                 selective_scan_backend='torch',scan_force_torch=True)
        original.eval(); standalone.eval()
        x = torch.randn(2,3,16,16,requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a,b = original(x),standalone(y)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().mean().backward(); b.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
        for (_,left),(_,right) in zip(original.named_parameters(),standalone.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)

    def test_existing_cnn_and_tiny_contracts_are_accepted_without_modifying_runs(self):
        for key in ('cnn_finetuned','vit_tiny'):
            namespace=self.namespaces[key]
            folder=ROOT/'results'/key/'seed_42_pretrained_finetune_all_layers_epochs_20'
            if not (folder/'run_config.json').exists():
                continue
            contract=json.loads((folder/'run_config.json').read_text())
            data={k:v for k,v in contract['data'].items()
                  if k not in ('support_code_sha256','model_configs_sha256')}
            with tempfile.TemporaryDirectory() as tmp:
                target=Path(tmp)
                (target/'run_config.json').write_text(json.dumps(contract))
                result=namespace['notebook_program_identity'](target,contract['config'],data)
                self.assertEqual(result,contract['data'])
                self.assertTrue((target/'notebook_execution.json').is_file())

    def test_resume_original_checkpoint_reproduces_uninterrupted_weights(self):
        namespace=self.namespaces['cnn_finetuned']
        notebook=nbformat.read(model_notebook('cnn_finetuned'),as_version=4)
        metadata=notebook.metadata['standalone']
        legacy_hashes=json.loads((ROOT/'results/cnn_finetuned/seed_42_pretrained_finetune_all_layers_epochs_20/run_config.json').read_text())['data']['support_code_sha256'] if (ROOT/'results/cnn_finetuned/seed_42_pretrained_finetune_all_layers_epochs_20/run_config.json').exists() else {
            p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'lab_code').glob('*.py')}
        identity={'dataset':'synthetic-only','support_code_sha256':legacy_hashes,
                  'model_configs_sha256':metadata['legacy_config_digest']}
        with tempfile.TemporaryDirectory() as tmp:
            whole=Path(tmp)/'whole'; resumed=Path(tmp)/'resumed'
            model,loaders,config,device=setup()
            original_runtime.fit(model,loaders,config,device,whole,identity,{'synthetic':True})
            expected=torch.load(whole/'last_checkpoint.pt',weights_only=True)
            model,loaders,config,device=setup()
            original_epoch=original_runtime.train_one_epoch
            calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise KeyboardInterrupt()
                return original_epoch(*args,**kwargs)
            with patch.object(original_runtime,'train_one_epoch',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    original_runtime.fit(model,loaders,config,device,resumed,identity,{'synthetic':True})
            new_identity=namespace['notebook_program_identity'](resumed,config,{'dataset':'synthetic-only'})
            model,loaders,config,device=setup()
            namespace['fit'](model,loaders,config,device,resumed,new_identity,{'synthetic':True})
            actual=torch.load(resumed/'last_checkpoint.pt',weights_only=True)
            self.assertEqual(actual['epoch'],2)
            self.assertEqual(actual['contract'],expected['contract'])
            for name,weight in expected['model'].items():
                torch.testing.assert_close(actual['model'][name],weight,rtol=0,atol=0)

    def test_single_copied_notebook_works_without_helper_files(self):
        path=model_notebook('vit_tiny')
        with tempfile.TemporaryDirectory() as tmp:
            isolated=Path(tmp); target=isolated/path.name
            target.write_bytes(path.read_bytes())
            before=Path.cwd()
            try:
                os.chdir(isolated)
                namespace=load_definitions(target)
                identity=namespace['notebook_program_identity'](isolated/'new_run',namespace['CONFIG'],{'dataset':'synthetic-only'})
                self.assertIn('notebook_code_sha256',identity)
                self.assertEqual(namespace['NOTEBOOK_PATH'],target)
                self.assertFalse((isolated/'lab_code').exists())
                self.assertFalse((isolated/'MODEL_CONFIGS.json').exists())
            finally:
                os.chdir(before)

    def test_notebook_changes_reject_resume(self):
        path=model_notebook('cnn_finetuned')
        namespace=self.namespaces['cnn_finetuned']
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            identity=namespace['notebook_program_identity'](folder,namespace['CONFIG'],{'dataset':'synthetic-only'})
            (folder/'run_config.json').write_text(json.dumps({'config':namespace['CONFIG'],'data':identity}))
            old=namespace['NOTEBOOK_PATH']
            edited=nbformat.read(path,as_version=4)
            next(c for c in edited.cells if c.metadata.get('lab_role')=='model').source+='\n# changed implementation\n'
            target=folder/path.name;nbformat.write(edited,target)
            try:
                namespace['NOTEBOOK_PATH']=target
                with self.assertRaisesRegex(ValueError,'Notebook code or data changed'):
                    namespace['notebook_program_identity'](folder,namespace['CONFIG'],{'dataset':'synthetic-only'})
            finally:
                namespace['NOTEBOOK_PATH']=old

    def test_final_test_is_after_selection_and_cached(self):
        namespace=self.namespaces['cnn_finetuned']
        model,loaders,config,device=setup()
        identity={'dataset':'synthetic-only'}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Complete training/validation'):
                namespace['evaluate_test'](model,model,folder,{},config,device,folder,identity,False)
            namespace['fit'](model,loaders,config,device,folder,identity,{'synthetic':True})
            partitions={'train':[{'label':0},{'label':1}], 'test':[{'label':0},{'label':1}]}
            original_dataset=namespace['layout_dataset']
            original_evaluate=namespace['evaluate']
            try:
                namespace['layout_dataset']=lambda *args:Samples(6)
                metrics=namespace['evaluate_test'](model,model,folder,partitions,config,device,folder,identity,False)
                self.assertEqual(metrics['records'],6)
                self.assertTrue((folder/'test_predictions.csv').exists())
                self.assertTrue((folder/'test_confusion_matrix.png').exists())
                def fail_if_repeated(*args,**kwargs):
                    raise AssertionError('Cached final test must not evaluate again')
                namespace['evaluate']=fail_if_repeated
                cached=namespace['evaluate_test'](model,model,folder,partitions,config,device,folder,identity,False)
                self.assertEqual(metrics,cached)
            finally:
                namespace['layout_dataset']=original_dataset
                namespace['evaluate']=original_evaluate


if __name__=='__main__': unittest.main()
