# Published fine-tuning results

Snapshot: 2026-10-19T00:43:01.215905+00:00

Each completed row reports the checkpoint selected by minimum validation loss. An in-progress row has no final test score. These are single-seed runs (seed 42); no result is inferred from a partial run.

| Model | Status | Epochs | Selected epoch | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [CNN fine-tuned (EfficientNet-B0)](cnn_finetuned/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 15/20 | 8 | 74.37% | 51.96% | 50.43% |
| [ResNet101](resnet101/seed_42_pretrained_finetune_all_layers_epochs_20) | in progress | 9/20 | — | — | — | — |
| [ResNet50](resnet50/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 15/20 | 8 | 56.37% | 53.42% | 51.67% |
| [ViT-Tiny](vit_tiny/seed_42_pretrained_finetune_all_layers_epochs_20) | completed | 14/20 | 7 | 56.60% | 56.26% | 56.87% |

## Included files

Run configuration and initialization provenance, history (JSON/CSV), training/validation curves, environment metadata, and snapshot status are included for every published run. Completed runs additionally include selected-checkpoint metrics, the completion marker, test metrics/predictions, and the test confusion matrix. The CSV summary also records balanced accuracy and macro-F1.

Checkpoints, ImageNet weights, dataset arrays, and full training/validation prediction CSVs remain local. Run-directory names retain the original epoch budget for checkpoint compatibility; the history and status report actual completed epochs.

In-progress files are a consistent snapshot of saved completed epochs. They do not update automatically when training continues. Plots are rendered from the captured history without running inference or training.
