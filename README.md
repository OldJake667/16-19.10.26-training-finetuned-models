# ImageNet-pretrained models: RDN full fine-tuning

Nine models initialized with published ImageNet weights and trained on the same compact RDN128 split as the [from-scratch experiment](https://github.com/OldJake667/training_models_12-16.10.26_from_scratch). All backbone layers and the new two-class heads train.

## Repository layout

| Folder or file | Contents |
| --- | --- |
| [`standalone_notebooks/`](standalone_notebooks) | One environment check and nine complete, self-contained model notebooks |
| [`published_results/`](published_results) | Saved metrics, learning curves, test predictions, and completed/in-progress status |
| [`verification/`](verification) | Notebook verification reports, dataset provenance, and upload manifest |
| [`licenses/`](licenses) | Attribution for the embedded VMamba implementation |
| `requirements.txt`, `start_jupyter.sh` | Dependencies and optional Jupyter launcher |

Only the self-contained notebooks are published. Older wrappers, archived notebook variants, preparation helpers, environments, datasets, and weight files remain outside the published repository.

## Models and settings

| Notebook | Backbone | Parameters | Maximum epochs |
| --- | --- | ---: | ---: |
| [01 CNN fine-tuned](standalone_notebooks/01_cnn_finetuned.ipynb) | EfficientNet-B0 | 4.01M | 20 |
| [02 ViT-Tiny fine-tuned](standalone_notebooks/02_vit_tiny_finetuned.ipynb) | ViT-Tiny/16, original grayscale architecture | 5.43M | 20 |
| [03 ResNet50 fine-tuned](standalone_notebooks/03_resnet50_finetuned.ipynb) | ResNet50 | 23.51M | 20 |
| [04 ResNet101 fine-tuned](standalone_notebooks/04_resnet101_finetuned.ipynb) | ResNet101 | 42.50M | 20 |
| [05 ResNet152 fine-tuned](standalone_notebooks/05_resnet152_finetuned.ipynb) | ResNet152 | 58.15M | 10 |
| [06 ViT-Small fine-tuned](standalone_notebooks/06_vit_small_finetuned.ipynb) | ViT-Small/16 | 21.67M | 20 |
| [07 ViT-Base fine-tuned](standalone_notebooks/07_vit_base_finetuned.ipynb) | ViT-Base/16 | 85.80M | 10 |
| [08 ViT-Large fine-tuned](standalone_notebooks/08_vit_large_finetuned.ipynb) | ViT-Large/16 | 303.30M | 10 |
| [09 VMamba-Base fine-tuned](standalone_notebooks/09_vmamba_base_finetuned.ipynb) | Official VMamba-Base s2l15 | 87.53M | 10 |

All models use early-stopping patience **7**, selected by minimum validation loss. The initial recipe uses AdamW (backbone `1e-5`, head `1e-4`, weight decay `0.05`), two warmup epochs, cosine decay, BF16, and effective batch size 128 through gradient accumulation. Physical batches and activation checkpointing are configured for the available memory. These recipes have not been optimized for RDN; pretraining does not guarantee improved accuracy.

EfficientNet, ResNets, and VMamba use ImageNet-1k sources. The ViTs use ImageNet-21k pretraining followed by ImageNet-1k adaptation. ViT-Tiny maps every pretrained backbone tensor into the original grayscale model. Exact model IDs and initialized-state hashes are recorded with each run.

The CNN changes architecture from the original 14.84M custom CNN to EfficientNet-B0; VMamba changes from Tiny to Base. Those comparisons change capacity or architecture as well as initialization. The other seven model parameter counts are preserved.

## Dataset and evaluation

| Split | Images | Share of all images |
| --- | ---: | ---: |
| Training | 97,200 | 87.57% |
| Validation | 10,800 | 9.73% |
| Held-out test | 3,000 | 2.70% |

The original 108,000 training images are split 90/10 into training and validation. The released 3,000-image test set stays separate. Every notebook embeds the four original file hashes and validates the split before training. Labels use the first `argmax(compact_response) >= 83`; this compact-response target is not an exact recreation of a raw-reflection 8.3 GHz target.

Training/validation loss and accuracy are plotted by epoch. Balanced accuracy, macro-F1, AUC, confusion matrices, and prediction counts are also recorded. The lowest-validation-loss checkpoint is selected before final test evaluation; test results do not select epochs or recipes.

## Published runs

See the [results table](published_results/README.md), [CSV summary](published_results/summary.csv), and [JSON summary](published_results/summary.json). Completed results and snapshots of ongoing runs are labeled separately. Empty test fields mean evaluation is not yet available. These are single-seed results (seed 42).

Completed runs currently include CNN fine-tuned, ViT-Tiny, and ResNet50. ResNet101 has a saved in-progress snapshot. The remaining model notebooks are prepared; no completed results are claimed for them. Histories and result snapshots are updated when explicitly published, rather than automatically during training.

## Run in Jupyter

On the configured computer:

```bash
cd /home/murat/Desktop/fine_tuned_models
./start_jupyter.sh
```

Open [00_environment_and_data_check.ipynb](standalone_notebooks/00_environment_and_data_check.ipynb), then one model notebook. Review the settings cell and use **Restart Kernel and Run All Cells**. Run one GPU model at a time and shut down its kernel before starting another. The launcher opens `standalone_notebooks/`.

For a fresh clone, create a Python environment with a compatible CUDA [PyTorch/torchvision installation](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
export RDN_DATA_ROOT=/path/to/rdn_package
# Optional directory containing the already downloaded VMamba-Base checkpoint:
export RDN_PRETRAINED_DIR=/path/to/pretrained
python -m ipykernel install --user --name rdn-finetune --display-name 'RDN fine-tuning'
python -m jupyter lab
```

The raw dataset directory must contain `data/full_manifest.csv`, `data/images.npy`, `data/labels.npy`, and `data/spectra.npy`. The notebook settings expose paths, a complete editable `CONFIG`, and training/test flags. No project helper scripts or external configuration JSON are loaded. Installed PyTorch/timm libraries supply the standard backbones; the custom grayscale Tiny model and VMamba implementation are embedded in their notebooks.

A first run obtains published weights; a compatible saved run resumes from its checkpoint without downloading them again. Save notebook edits before starting. Changing settings or implementation requires a new `RUN_NAME`. The original checkpoint contracts can be reused only when their settings and provenance match the embedded verified compatibility snapshot. `published_results/` is an archive for viewing; live local training continues to use the ignored `results/` directory.

VMamba requires the supported Mamba-1 `selective_scan_cuda` extension and a compatible CUDA compiler/PyTorch setup. Follow the [official Mamba installation instructions](https://github.com/state-spaces/mamba/tree/v2.2.6.post3) and enable the optional kernel check in the environment notebook. The official Base checkpoint is verified against SHA-256 `61aec3c5a67cc351dbb8a04c0a3ff76de213e01ff387cc349f94e497ce377452`.

## Verification

The published reports record architecture equivalence, strict pretrained weight loading, synthetic checkpoint resume, and test-cache checks. CI validates the self-contained notebooks and published result consistency without starting training or using dataset images. The notebook refactor preserves the existing recipes; it does not establish a solution to the earlier ViT convergence problems. Checkpoints, pretrained weights, raw data, credentials, and full training/validation prediction CSVs are excluded from GitHub.
