# Grad-CAM on wrong predictions

## Published results

This is the **original bandwidth experiment**, separate from RDN fine-tuning.
The saved run contains six wrong predictions across CNN, ViT-Tiny, DINOv2,
and VMamba-Tiny. All six are true Low / predicted High. There are no false-Low
cases in these checkpoints' validation predictions.

- [Browse the six comparisons](published_results/README.md)
- [Executed Jupyter notebook](10_gradcam_errors.ipynb)
- [Verification report](VERIFICATION.json)
- [Additional DINOv2 image-covering checks](structural_checks/README.md)

`published_results/53bd12555dbda8d0/` preserves the run used in the presentation:
original images, predicted/true-class overlays, raw maps, prediction audits,
and metadata. Absolute paths in the archived metadata identify the original
execution location; use the relative links here to browse this upload.

## Run in Jupyter

Open `10_gradcam_errors.ipynb` in Jupyter, select the existing PyTorch kernel,
set `PACKAGE_ROOT`, and run the cells in order. The notebook embeds the full
Grad-CAM implementation. It uses the existing original bandwidth XAI package's
`xai_code`, validation BMPs, saved checkpoints, and archived predictions.
No training or weight download is performed.

`PACKAGE_ROOT` must be the original folder containing `data/`, `checkpoints/`,
`reference/`, and `xai_code/`, previously used for LIME and gradient attention
rollout. It is **not** the full RDN training data folder. You may set the
`BANDWIDTH_XAI_ROOT` environment variable instead. On the current Mac the
settings default to the existing local package. On another machine edit the
path in the settings cell.

The default selection includes CNN, ViT-Tiny, DINOv2, and VMamba-Tiny. Every
model first audits all 45 validation predictions against the archived results.
Changed labels or probability drift above 0.0005 stop execution. All mistakes
are then explained, without confidence filtering or choosing attractive maps.
Both error directions are handled automatically. The current checkpoints have
six model-image errors across three unique images, **all true Low / predicted
High**; there are no true High / predicted Low cases in this validation set.

Each case shows the original image, Grad-CAM for the wrong predicted-class
logit, and Grad-CAM for the true-class logit. Probability is displayed for the
prediction, but gradients target pre-softmax logits. Native signed CAMs,
positive CAMs, normalized maps, empty-map flags, raw logits, input/checkpoint
hashes, target layer, and unchanged-weight verification are saved. Each class
is normalized separately, so its colors cannot be compared as absolute
amounts of evidence. An empty positive map is preserved rather than fabricated.

Target layers: the CNN's last convolution; the final block's pre-attention
`norm1` for ViT-Tiny and DINOv2 (CLS token removed); and VMamba's spatial
classifier norm before global pooling. Transformer and VMamba results are
adaptations of the Grad-CAM calculation to spatial feature maps, not
attention rollout. Frozen weights do not stop attribution: a local gradient
leaf at the target activation supplies the exact downstream logit derivative.
The hook checks unchanged logits and is always removed.

CUDA with the existing Mamba-1 scan extension is recommended for VMamba. The
explicit `ALLOW_VMAMBA_CPU_REFERENCE` setting enables the slower original
PyTorch reference scan; the current Mac's outputs use that path and reproduce
the archived probabilities within the stated tolerance. No MPS path is used.

The original split is **176 training / 45 validation / no independent test**.
Checkpoints were selected using these validation images. These explanations
are descriptive, not independent test estimates or proof of physical causality.
They must not be mixed with the RDN resonance-peak classification experiments.

Outputs go to `results/<configuration_id>/`. Rerunning recomputes the audit and
all errors; existing files in that configuration are regenerated. Change
`OUTPUT_ROOT` to keep a separate copy. The selected image identities and all
error-direction counts are recorded. The workflow fails rather than replacing
missing checkpoints with untrained or unrelated weights.

Dependencies: the existing environment's `torch`, `torchvision`, `timm`,
`numpy`, `Pillow`, and `matplotlib`; no additional Grad-CAM library is required.

Method: [Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
Transformer layer selection: [pytorch-grad-cam author documentation](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md).

`workflow.py` is an equivalent module for scripted execution. Run
`python verify_gradcam.py` for the analytical-gradient, empty-map, patch-order,
hook-cleanup, and error-direction checks.
