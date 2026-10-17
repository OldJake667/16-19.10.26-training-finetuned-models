# 16–19 October 2026: fine-tuned RDN models

Fine-tune published pretrained models on the **same compact RDN128 data and split** as the earlier from-scratch experiment. The default is **full fine-tuning**: all backbone layers and a new two-class head are trainable. This repository contains runnable code and clean notebooks, not completed training results.

Previous experiment: [12–16 October, from scratch](https://github.com/OldJake667/training_models_12-16.10.26_from_scratch).

## Which weights should I use?

Use **published pretrained weights** for the main experiment. This tests whether transfer learning helps compared with random initialization. Keep your old validation-selected checkpoints as baselines. Continuing those checkpoints is a separate experiment in additional training, not ImageNet initialization.

Two exceptions need explicit labels:

- The original custom three-block CNN has **no published ImageNet checkpoint**. `cnn_resnet18` is a separately labelled ImageNet-pretrained CNN reference with a different architecture. `cnn` optionally continues the exact custom CNN from a supplied checkpoint.
- `dinov2` uses the published DINOv2 pretrained backbone, whose source training data differs from the ImageNet supervised models. It is an additional reference: DINOv2 was excluded from the nine active RDN scratch runs.

## Quick start on the lab computer

Use the existing working CUDA/PyTorch environment from the scratch experiment, or install a compatible torch/torchvision pair using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). An NVIDIA GPU and Linux/WSL2 are recommended; VMamba requires its CUDA scan extension. The locally checked CPU versions are torch 2.8.0 / torchvision 0.23.0; the CUDA lab must pass the preflight independently.

```bash
git clone https://github.com/OldJake667/16-19.10.26-training-finetuned-models.git
cd 16-19.10.26-training-finetuned-models
# Activate your existing CUDA virtual environment here.
python -m pip install -r requirements.txt

# Root of the EXISTING RDN package, containing data/ and PACKAGE_MANIFEST.json:
export RDN_DATA_ROOT=/path/to/existing/rdn_package
python tools/preflight.py --data-root "$RDN_DATA_ROOT"

# A practical first fine-tuning run:
python train.py --model resnet50 --data-root "$RDN_DATA_ROOT"
```

**Original data and trained weight files are not uploaded here.** Reuse the full RDN package already on your lab machine. The loader verifies the exact four data-file SHA-256 values, target, unique geometries, and original split. `PACKAGE_MANIFEST.json` in this repository records the required data identities. Point `--data-root` to the existing package; do not download a different split for comparison.

### Jupyter notebooks

```bash
python -m ipykernel install --user --name rdn-finetune --display-name "RDN fine-tuning"
python -m jupyter lab
```

Open `notebooks/00_environment_and_data_check.ipynb`, select **RDN fine-tuning**, set `DATA_ROOT`, and run it. Then open a model notebook, review the configuration, and use **Run All**. Each notebook has explicit `RUN_TRAINING` and `RUN_FINAL_TEST` switches. There is no training or download at import time. Run one model at a time and shut down its kernel before starting another.

```bash
# Run all ten pretrained references sequentially (potentially a long GPU job):
python tools/run_all.py --data-root "$RDN_DATA_ROOT"
# Or start with a smaller group:
python tools/run_all.py --data-root "$RDN_DATA_ROOT" --models resnet50 vit_tiny vit_small
# Inspect settings without loading data or weights:
python train.py --model vit_tiny --print-config
```

## Models and initial weights

| Key | Initialization | Physical batch / accumulation | Comparison note |
| --- | --- | --- | --- |
| `cnn_resnet18` | `resnet18.a1_in1k` | 32 / 4 | Additional CNN reference; not the original custom CNN |
| `vit_tiny` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 16 / 8 | Weights mapped to original grayscale ViT-Tiny |
| `resnet50` | `resnet50.a1_in1k` | 16 / 8 | Same architecture family as scratch |
| `resnet101` | `resnet101.a1h_in1k` | 8 / 16 | Same architecture family as scratch |
| `resnet152` | `resnet152.a1h_in1k` | 4 / 32 | Same architecture family as scratch |
| `vit_small` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 16 / 8 | ImageNet-21k pretraining, ImageNet-1k fine-tuned source |
| `vit_base` | `vit_base_patch16_224.augreg_in21k_ft_in1k` | 4 / 32 | Same source family |
| `vit_large` | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 1 / 128 | Activation checkpointing; profile memory first |
| `dinov2` | `vit_small_patch14_dinov2.lvd142m` | 8 / 16 | Additional self-supervised pretrained reference |
| `vmamba` | official `vmamba_tiny_s1l8` ImageNet weights | 1 / 128 | CUDA scan and activation checkpointing |
| `cnn` | User-supplied original trained checkpoint | 32 / 4 | Optional continuation only; excluded from pretrained `run_all` |

For grayscale ViT-Tiny, all pretrained backbone tensors are mapped to the original `nn.MultiheadAttention` architecture. RGB patch kernels are summed to obtain the one-channel projection, and pretrained LayerNorm epsilon is retained. The task head is new. A synthetic equivalence test checks that its evaluation outputs match the corresponding RGB model with replicated grayscale inputs. No incompatible parameters are silently skipped.

Weight downloads occur on the first new run. timm model IDs and a SHA-256 of the actual initial state (including the seeded task head) are saved in `model_info.json`. VMamba additionally requires an exact checkpoint hash. Download failures stop the run; they never cause a random-initialization fallback. Resume/evaluation-only mode rebuilds the architecture and loads the saved checkpoint without re-downloading pretrained weights.

## Training recipe and comparison controls

- Data: 111,000 compact layouts; **97,200 train / 10,800 validation / 3,000 held-out test**. Class 0: first `argmax(compact_response) < 83`; class 1: `>= 83`. This is not an exact reproduction of raw-reflection 8.3 GHz labels.
- Defaults: seed 42; maximum 20 epochs; early-stopping patience 5; two warmup epochs followed by cosine decay. No inherited five-epoch caps.
- AdamW: backbone LR `1e-5`, head LR `1e-4`, weight decay `0.05` with no decay for bias/norm/token embeddings. These are starting recipes, not validated optimal settings.
- Effective batch 128 via sample-weighted accumulation, including partial windows. Physical batches differ; BatchNorm statistics therefore differ too. Smaller batches are starting points for the 8 GB GPU, not a guarantee that every model fits.
- All layers train. No feature caching, frozen-head-only training, random crop, flip, rotation or erasing. The full geometry is resized. Input normalization follows the weight family; interpolation and channel count are explicitly logged. Most timm models use bicubic resizing; custom CNN, grayscale ViT-Tiny and VMamba retain bilinear resizing.
- The checkpoint with **minimum validation loss** is selected. Held-out test evaluation happens only after selection; `--no-test` disables it. No test result influences training or selection.
- Every validation epoch logs balanced accuracy, macro-F1 and predicted class counts. A one-class-prediction warning makes collapse visible; it does not change the recipe or choose a checkpoint using test data. Final metrics include accuracy, balanced accuracy, macro-F1, ROC-AUC and a confusion matrix.

Compare with the saved scratch baselines, accounting for their unequal epoch budgets and any architecture/preprocessing differences. These runs test transfer; improvement is not guaranteed.

## Resume, evaluation, and outputs

Outputs are separate from the old experiment: `results/<model>/seed_42_pretrained_finetune_all_layers_epochs_20/`.

Each run saves configuration and data identity, source code hashes, actual initial-weight identity, total/trainable parameter counts, environment, last/best checkpoints, optimizer/scheduler/scaler/RNG state, epoch history, train/validation metrics and predictions, and accuracy/loss plots in PNG. Optional vector PDF export is available through `lab_code.runtime.plot_history(..., export_pdf=True)`. Final evaluation adds `test_metrics.json`, `test_predictions.csv`, and `test_confusion_matrix.png`.

Re-running the same command resumes at the last complete epoch. An interrupted partial epoch is repeated. Completed runs load the validation-selected checkpoint. Changing settings, data or source code requires a **new run name**. Use `--eval-only` with the same model/config/run name to load a completed run. Test outputs are cached against the selected checkpoint hash.

```bash
# Explicit new recipe; preserve effective batch 128 when reducing the physical batch:
python train.py --model vit_base --data-root "$RDN_DATA_ROOT" \
  --batch-size 2 --accumulation-steps 64 --run-name seed42_batch2_finetune
python tools/summarize_results.py

# Optional original-CNN continuation (not ImageNet pretraining):
python train.py --model cnn --data-root "$RDN_DATA_ROOT" \
  --checkpoint /path/to/original/cnn/best_checkpoint.pt
```

Checkpoint continuation loads model tensors strictly, starts a new optimizer/schedule, and records the source checkpoint hash/epoch. It does not resume the old optimizer or copy old reported scores. Continuation of other architectures is available via `--initialization checkpoint --checkpoint ...` when the exact two-class state dict matches.

## VMamba setup

Reuse the working `selective_scan_cuda` installation from the scratch environment. Check both forward and backward before fine-tuning:

```bash
python tools/preflight.py --data-root "$RDN_DATA_ROOT" --vmamba-kernel
# Reuse the already verified original ImageNet file if present:
python train.py --model vmamba --data-root "$RDN_DATA_ROOT" \
  --weights-dir /path/to/original/package/pretrained
```

Otherwise the official checkpoint is downloaded into ignored `weights/` and checked against SHA-256 `d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a`.

For a new CUDA environment, follow the [official Mamba installation/build instructions](https://github.com/state-spaces/mamba/tree/v2.2.6.post3) and the already-working toolchain notes in the old RDN package. The vendored implementation uses the Mamba-1 `selective_scan_cuda` interface; installing an unrelated Mamba-2 kernel is insufficient. Its source and MIT notice are retained in `licenses/VMamba_LICENSE.txt`. GPU/kernel execution and memory capacity must be checked on the target machine.

## Checks and provenance

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
python tools/verify_models.py --data-root "$RDN_DATA_ROOT"
# Optional real published-weight loading check; downloads the small ViT-Tiny weights:
python tools/verify_models.py --pretrained-tiny
```

Tests use synthetic data and temporary output folders. See `VERIFICATION.json` for checks actually completed locally. No research training or held-out model evaluation is claimed here.

The data/runtime and original custom architectures derive from commit `380a36849e292526cdd99a928e123c3abae4d755` of the scratch repository. The original dataset and model checkpoint files remain outside this code repository. Third-party model weights retain their upstream licenses; weights are not redistributed here.

Primary references: [PyTorch fine-tuning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [timm](https://github.com/huggingface/pytorch-image-models), [ViT AugReg weights](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k), [DINOv2 weights](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m), [VMamba](https://github.com/MzeroMiko/VMamba).
