# 16–19 October 2026: fine-tuned RDN models

Nine ImageNet-pretrained models are configured for **full fine-tuning** on the same compact RDN128 data and fixed split as the [from-scratch experiment](https://github.com/OldJake667/training_models_12-16.10.26_from_scratch). All backbone layers and the new two-class heads train. These are prepared experiments; no completed research training results are claimed.

## Selected models

| Display name | Model key | Pretrained architecture | Physical batch / accumulation |
| --- | --- | --- | --- |
| **CNN fine-tuned** | `cnn_finetuned` | EfficientNet-B0, `efficientnet_b0.ra_in1k` | 16 / 8 |
| ViT-Tiny | `vit_tiny` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k`, mapped to the original grayscale architecture | 16 / 8 |
| ResNet50 | `resnet50` | `resnet50.a1_in1k` | 16 / 8 |
| ResNet101 | `resnet101` | `resnet101.a1h_in1k` | 8 / 16 |
| ResNet152 | `resnet152` | `resnet152.a1h_in1k` | 4 / 32 |
| ViT-Small | `vit_small` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 16 / 8 |
| ViT-Base | `vit_base` | `vit_base_patch16_224.augreg_in21k_ft_in1k` | 4 / 32 |
| ViT-Large | `vit_large` | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 1 / 128 |
| VMamba-Base | `vmamba_base` | Official `vmamba_base_s2l15` | 1 / 128 |

**CNN fine-tuned** uses EfficientNet-B0. The original custom three-block CNN has no published ImageNet weights, so its comparison changes both architecture and initialization. VMamba-Base has **87,534,850 trainable parameters** with its two-class head, versus approximately 89 million with the original ImageNet head. Replacing the earlier Tiny scratch baseline with Base also changes capacity as well as initialization; retain that distinction when reporting results.

ViT-Tiny keeps the original grayscale architecture. Every pretrained backbone tensor is transferred explicitly, RGB patch kernels are summed into a one-channel projection, and the pretrained LayerNorm epsilon is retained. The task head is new. Synthetic checks verify equivalence to the corresponding RGB model receiving replicated grayscale inputs.

**ImageNet supplies the initial weights; RDN supplies the fine-tuning data.** The ViT weights were pretrained on ImageNet-21k and adapted to ImageNet-1k before this experiment; EfficientNet, ResNets, and VMamba use ImageNet-1k sources. Record this difference when comparing models. ImageNet images are not required locally. There is no linear probing, feature caching, or silent random-weight fallback.

All nine models are active in the default run list and self-contained notebooks. ViT-Tiny, ResNet50, and ViT-Base have been restored. The original scratch models and results remain in their separate project; `archived_notebooks/` retains older fine-tuning scaffold versions as historical references.

ImageNet is the baseline pretraining choice. For possible alternatives such as DINOv3, see [pretraining options](PRETRAINING_OPTIONS.md); which source works best on RDN must be established on the validation split.

## Run in Jupyter

On the configured computer:

```bash
cd /home/murat/Desktop/fine_tuned_models
./start_jupyter.sh
```

The local launcher uses the existing CUDA Python environment and RDN dataset, selects the **RDN fine-tuning** kernel, and opens `standalone_notebooks/`. Its machine-specific settings, environment, Jupyter runtime, and datasets are ignored by Git. Open `standalone_notebooks/00_environment_and_data_check.ipynb`, then one of the nine model notebooks in that folder. Every model filename ends in `_finetuned.ipynb`, for example `02_vit_tiny_finetuned.ipynb`. Review the configuration and select **Run All**. Run one model at a time and shut down its kernel before starting another.

For another computer, activate a compatible CUDA torch/torchvision environment using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
export RDN_DATA_ROOT=/path/to/existing/rdn_package
# Optional local directory containing the verified VMamba-Base ImageNet file:
export RDN_PRETRAINED_DIR=/path/to/pretrained
python tools/preflight.py --data-root "$RDN_DATA_ROOT"
python -m ipykernel install --user --name rdn-finetune --display-name 'RDN fine-tuning'
python -m jupyter lab
```

Alternatively, set `RDN_PYTHON` to your existing environment's Python executable and use `./start_jupyter.sh` with `RDN_DATA_ROOT` exported. The absolute path in the first example describes this local checkout; a clone can be placed elsewhere.

The self-contained notebooks expose `PROJECT_DIR`, `DATA_ROOT`, `WEIGHTS_DIR`, a complete editable `CONFIG` dictionary, `RUN_NAME`, `RUN_TRAINING`, and `RUN_FINAL_TEST`. All data loading, model construction, optimizer, training, checkpoint resume, validation, plotting, and test evaluation code is inside each notebook. They import installed libraries only and do not read project helper scripts, `MODEL_CONFIGS.json`, or `PACKAGE_MANIFEST.json`. The original data-file hashes are embedded. A single notebook can be copied elsewhere with the raw dataset and installed packages; adjust the paths in its settings cell.

The longer VMamba notebook includes the official architecture, scan operations, CUDA bindings, and MIT license. Standard ResNet, ViT, and EfficientNet backbones are constructed using installed timm; ViT-Tiny's original grayscale architecture and weight-transfer code are included explicitly. Compiled dependencies and pretrained weight files remain external packages/data.

Training and downloads occur only in the explicit launch cell. The first new run obtains published weights; resume rebuilds the architecture and loads its saved checkpoint without downloading them again. Notebook implementation code is hashed separately from editable settings and outputs. Save notebook edits before starting. Existing unchanged runs are accepted only when their configuration, data, initialization, and recorded source identities match the verified compatibility snapshot; `notebook_execution.json` then records the notebook identity while retaining the original checkpoint contract.

The older `notebooks/` folder remains intact so currently open notebooks and their outputs are preserved. Shut down the old GPU kernel before opening a self-contained replacement for the same model. The command-line tools remain available separately. This conversion preserves the model recipes, including current ViT-Tiny preprocessing and dropout; it does not claim to fix the diagnosed ViT convergence problems.

## Training and evaluation

- Fixed split: **97,200 training / 10,800 validation / 3,000 held-out test**. Labels use the first `argmax(compact_response) >= 83`, not an exact raw-reflection 8.3 GHz task. The four original data-file hashes are embedded in the self-contained notebooks and checked before training; the command-line path also pins them in `PACKAGE_MANIFEST.json`.
- Maximum epochs: **10** for ResNet152, ViT-Base, ViT-Large, and VMamba-Base; **20** for CNN fine-tuned, ViT-Tiny, ResNet50, ResNet101, and ViT-Small. All nine use early-stopping patience **7**, based on validation loss, with two warmup epochs and cosine decay over their configured budget. These settings are independent of scratch runs. Adjust the configuration before starting a new run.
- AdamW backbone learning rate `1e-5`, head learning rate `1e-4`, weight decay `0.05`, with no decay on bias/norm/token embeddings. These are starting recipes, not optimized settings.
- Effective batch **128** through sample-weighted accumulation. Physical batches are initial settings for the 8 GB GPU; full GPU memory/throughput checks remain necessary. ViTs and VMamba use activation checkpointing. EfficientNet and ResNets avoid activation checkpointing across BatchNorm layers.
- Full geometry is resized without random crops, flips, rotations, or erasing. Normalization and interpolation follow the recorded configuration. Physical batches differ, so BatchNorm statistics also differ.
- Minimum validation loss selects the checkpoint. Training/validation accuracy and loss plots appear in Jupyter and are saved. Each epoch also logs balanced accuracy, macro-F1, prediction counts, and one-class-prediction diagnostics.
- Final held-out evaluation saves accuracy, balanced accuracy, macro-F1, ROC-AUC, predictions, and a confusion matrix. Test scores do not select epochs or recipes. Set `RUN_FINAL_TEST=False` while developing a recipe.

The existing scratch ViT-Large continuation is a separate experiment. Preparing this repository does not change or start that run. Unequal budgets, pretrained data, and the CNN architecture change must be considered in comparisons; improved RDN performance is not guaranteed.

## Command line, resume, and outputs

```bash
python train.py --model resnet101 --data-root "$RDN_DATA_ROOT"
python train.py --model vmamba_base --data-root "$RDN_DATA_ROOT" --weights-dir "$RDN_PRETRAINED_DIR"
python train.py --model vit_large --print-config
# Sequential execution of the nine selected models; starts real training:
python tools/run_all.py --data-root "$RDN_DATA_ROOT"
python tools/summarize_results.py
```

Outputs use `results/<model>/seed_42_pretrained_finetune_all_layers_epochs_<max_epochs>/`, ending in `epochs_10` or `epochs_20` according to the model configuration. Each run records its configuration, data/source identities, exact initialized-state hash, total/trainable parameters, and software environment. It saves best/last checkpoints, optimizer/scheduler/scaler/RNG state, history, train/validation metrics and predictions, and plots. Optional vector plots are supported by `lab_code.runtime.plot_history(..., export_pdf=True)`.

An interrupted partial epoch is repeated; completed epochs resume from the last checkpoint. A completed run loads its validation-selected checkpoint. Changes to settings, data, or source code require a **new run name**. `--eval-only` loads a completed run, and `--no-test` disables held-out evaluation. Data, environments, pretrained weights, and trained checkpoints remain outside Git.

## VMamba-Base weights and kernel

The loader strictly checks the complete official pretrained backbone before replacing the original classifier. File: `vmamba_base_s2l15_imagenet.pth`; SHA-256:

```text
61aec3c5a67cc351dbb8a04c0a3ff76de213e01ff387cc349f94e497ce377452
```

The already downloaded file can be reused through `RDN_PRETRAINED_DIR` or `--weights-dir`. Otherwise it is downloaded into ignored `weights/` from the [official VMamba-Base release](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth).

Reuse the working Mamba-1 `selective_scan_cuda` extension and verify forward/backward before training:

```bash
python tools/preflight.py --data-root "$RDN_DATA_ROOT" --vmamba-kernel
```

A new environment needs the [official Mamba installation/build instructions](https://github.com/state-spaces/mamba/tree/v2.2.6.post3) and a matching CUDA/PyTorch compiler setup. An unrelated Mamba-2 kernel is insufficient. The vendored VMamba source retains its MIT notice under `licenses/`.

## Validation and provenance

```bash
MPLBACKEND=Agg python -m unittest tests.test_standalone_notebooks -v
# Original command-line implementation checks (clean legacy notebooks required):
MPLBACKEND=Agg python -m unittest tests.test_training -v
python tools/verify_models.py --data-root "$RDN_DATA_ROOT"
# Optional published EfficientNet weights and synthetic CPU forward check:
python tools/verify_models.py --pretrained-cnn
```

`VERIFICATION_STANDALONE.json`, `VERIFICATION.json`, and `VERIFICATION_PRETRAINED.json` describe checks actually performed. Synthetic checks are not research training or held-out evaluation. GPU profiling and convergence remain to be measured when the GPU is available.

The original data/runtime and custom architectures derive from scratch commit `380a36849e292526cdd99a928e123c3abae4d755`. Third-party weights retain their upstream licenses and are not redistributed here.

References: [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [timm model library](https://github.com/huggingface/pytorch-image-models), [ViT AugReg models](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k), [VMamba](https://github.com/MzeroMiko/VMamba).

## Notebook maintenance

The runnable notebooks contain their own code. Maintainers can rebuild clean copies with `python tools/build_standalone_notebooks.py`; this build tool is not required to run them. Back up notebook edits and outputs before regenerating. `MODEL_CONFIGS.json` remains the command-line/build-time configuration; editing a standalone notebook's `CONFIG` changes that notebook's run only.
