# DINOv2 image-covering checks

Additional forward passes through the original frozen DINOv2 classifier; no training.
Cover black pixels with white at the upper-right end, then compare with the same
number removed at the lower left. The exact masks, originals, protocol, and
probabilities are included. Native-image mean brightness is matched between the
two removals within each image; resizing may introduce small differences.

| Image | Original High | Upper-right removal | Lower-left removal |
| --- | ---: | ---: | ---: |
| M93 | 94.46% | 92.28% | 95.35% |
| M95 | 55.87% | 29.38% | 36.92% |

M93 stays High under both edits. M95 changes to Low under both edits; the result
does not isolate the upper-right shape as the only cause. Physical bandwidth was
not simulated for the edited images, so a changed prediction is not a confirmed
correction. M93 and M95 are distinct original designs, not before/after pairs.

See `structural_mask_checks.json` for the full protocol. To reproduce, run
`python reproduce_mask_checks.py --help` and supply `--package-root /path/to/original_bandwidth_xai`. The included masks
and originals are found next to the script automatically. The original checkpoints
are required.
