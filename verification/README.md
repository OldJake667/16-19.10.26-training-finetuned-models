# Verification and provenance

The JSON reports describe checks actually performed while preparing the models and self-contained notebooks. They are historical verification snapshots, not training scores. Earlier reports may refer to local preparation helpers that are intentionally excluded from this repository; the runnable implementations are fully embedded in `standalone_notebooks/`.

`data_manifest.json` records the four dataset hashes and fixed split; the same hashes are embedded in every model notebook. `UPLOAD_MANIFEST.json` lists the paths, sizes, and SHA-256 hashes in the published repository snapshot, excluding itself. Checkpoint contracts in published run configurations retain their original provenance, including older source hashes, so saved-run identity is not rewritten during publication.

`VMAMBA_CHECKPOINT_DIAGNOSTIC.json` records the checkpoint-warning investigation. The reduced CPU model had gradients for all trainable parameters; the no-gradient-input warning was reproduced during validation. Early and late backbone tensors in the actual epoch-4 checkpoint differed from initialization. A proposed cleanup was tested separately and was not applied to the completed training run.
