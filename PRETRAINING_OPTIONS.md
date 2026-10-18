# Pretraining choices for the RDN comparison

**Pretraining** supplies the initial backbone weights. **Fine-tuning** trains
those weights and a new two-class head on the RDN training split. ImageNet
images are not required to fine-tune the downloaded checkpoints.

## Current baseline: supervised ImageNet weights

The nine active notebooks use ImageNet-pretrained weights. EfficientNet-B0,
ResNets, and VMamba-Base use ImageNet-1k sources. ViT-Tiny/Small/Base/Large use
ImageNet-21k pretraining followed by ImageNet-1k adaptation. These are already
different pretraining data budgets; record the exact model IDs rather than
treating them as identical sources. The model IDs and initialized-state hashes
are saved with each run.

This baseline provides a direct starting point for comparing with the original
scratch experiment. The CNN reference changes to EfficientNet-B0 and VMamba
changes from Tiny to Base, so those comparisons also change architecture or
capacity. No source is guaranteed to produce better RDN validation results.

## Possible additional reference: DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) is a newer self-supervised
family with web-image-pretrained ViT-S/16 (21M parameters) and ViT-B/16 (86M),
among other models. Those checkpoints use the LVD-1689M corpus rather than
ImageNet supervised pretraining. The [official model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
describes the data, model variants, and evaluation results.

ViT-S/16 is a reasonable smaller candidate for an additional laptop experiment;
this is a hardware-oriented choice, not a measured RDN performance claim. It
would need its own checkpoint loader, preprocessing, GPU profile, new task head,
and separate run/initialization label. It has not been added automatically to
the nine-model ImageNet comparison.

## Domain adaptation

A separate research option is self-supervised adaptation using only unlabeled
RDN training layouts before supervised fine-tuning. Domain matching could help,
but improvement is an experimental hypothesis and adds compute. Keep validation
and test images out of that adaptation stage; compare recipes on validation,
then evaluate the selected recipe on the held-out test set.

Recommendation: complete the nine ImageNet-initialized baselines first, then
consider DINOv3 ViT-S/16 as a separately labelled additional experiment. Keep
splits, evaluation, and compute-budget reporting consistent.
