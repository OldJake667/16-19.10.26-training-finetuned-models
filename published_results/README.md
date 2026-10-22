# Published fine-tuning results

Snapshot: 2026-10-22T21:56:54.819480+00:00

Completed rows evaluate the checkpoint selected by minimum validation loss. In-progress snapshots have no final test score. These are single-seed runs (seed 42).

| Model | Status | Epochs | Selected epoch | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [CNN fine-tuned (EfficientNet-B0)](cnn_finetuned/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 15/20 | 8 | 74.37% | 51.96% | 50.43% |
| [ResNet101](resnet101/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 20/20 | 19 | 54.93% | 53.20% | 53.10% |
| [ResNet152](resnet152/seed_42_pretrained_finetune_all_layers_epochs_10) | completed | 10/10 | 10 | 53.94% | 53.25% | 53.73% |
| [ResNet50](resnet50/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 15/20 | 8 | 56.37% | 53.42% | 51.67% |
| [ViT-Tiny](vit_tiny/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 14/20 | 7 | 56.60% | 56.26% | 56.87% |
| [VMamba-Base](vmamba_base/seed_42_pretrained_finetune_all_layers_epochs_10) | completed | 10/10 | 9 | 68.73% | 66.68% | 68.27% |

## Included files

Every run includes its configuration, initialization provenance, history (JSON/CSV), training/validation curves, environment metadata, and snapshot status. Completed runs also include selected-checkpoint metrics, test predictions, test metrics, and the confusion matrix. Balanced accuracy and macro-F1 are recorded in the summaries.

Where a self-contained notebook has saved execution outputs, its exact saved copy is included in that run directory. These copies document completed execution and are for viewing; use the clean notebooks in `../standalone_notebooks/` to start a run. Older helper-based wrappers are not published.

Plots are rendered from the captured history without inference or training. Checkpoints, pretrained weights, raw data, and full training/validation prediction CSVs remain local. Run directory names retain their original configured epoch budgets; histories record actual completed epochs. Snapshots update only when explicitly published.
