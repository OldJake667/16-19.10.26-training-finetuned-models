# Grad-CAM: six wrong predictions

Original bandwidth validation set: 45 images per model; all errors shown.
Each comparison shows the original image, the wrong High prediction, and the true Low class.

| Model | Image | True | Predicted | High probability | Figure |
| --- | --- | --- | --- | ---: | --- |
| CNN | M93 | Low | High | 96.35% | [View](53bd12555dbda8d0/cnn/M93_log-20241114-212937/comparison.png) |
| ViT-Tiny | M323 | Low | High | 57.55% | [View](53bd12555dbda8d0/vit_tiny/M323_log-20241114-212937/comparison.png) |
| ViT-Tiny | M93 | Low | High | 87.75% | [View](53bd12555dbda8d0/vit_tiny/M93_log-20241114-212937/comparison.png) |
| DINOv2 | M93 | Low | High | 94.46% | [View](53bd12555dbda8d0/dinov2/M93_log-20241114-212937/comparison.png) |
| DINOv2 | M95 | Low | High | 55.87% | [View](53bd12555dbda8d0/dinov2/M95_log-20241114-212937/comparison.png) |
| VMamba-Tiny | M93 | Low | High | 93.25% | [View](53bd12555dbda8d0/vmamba/M93_log-20241114-212937/comparison.png) |

Bright colors show stronger positive Grad-CAM for that class. Each class map is scaled separately, so colors do not compare absolute evidence between Low and High. ViT-Tiny has an empty positive High map for both errors; those empty maps are retained.

[CSV summary](53bd12555dbda8d0/summary.csv) · [JSON summary](53bd12555dbda8d0/summary.json)

Model weights were unchanged. These images were used for checkpoint selection; this is descriptive validation analysis, not independent test performance or proof of physical causes.
