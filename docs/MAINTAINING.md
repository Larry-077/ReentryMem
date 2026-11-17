# Maintenance Notes

The main README introduces the research and the inference workflow. This page records release and development details.

## Validation

The standalone release has passed 17 CPU tests covering input contracts, model forwards, entity-attention hooks, mixed precision, sampling updates, video I/O, and upload recovery. The complete single-entity checkpoint has passed strict loading of all 829 tensors. A small forward pass using the real weights and calibrated region hooks also passed with synthetic text embeddings.

Checkpoint checksums and CLI preflight passed. Full-resolution CUDA video generation has not yet been re-run for this standalone package; the CPU checks do not cover the complete T5/VAE generation chain.

```bash
python -m unittest discover -s tests -v

python infer.py \
  --case examples/case.json \
  --checkpoint checkpoints/ReentryMem \
  --base-model pretrained/Wan2.2-TI2V-5B \
  --dry-run --verify-checkpoint
```

## Export a Checkpoint

```bash
python tools/export_checkpoint.py \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/ReentryMem-model \
  --name ReentryMem --variant single-entity --step 1496
```

The bundle contains merged DiT weights, architecture configuration, region calibration, inference defaults, the model card, licenses, and SHA256 checksums. `--hardlink` stages local weights without another copy; keep linked weights unchanged. Optimizer state and training data are excluded.

The model card is rendered from [MODEL_CARD.md](MODEL_CARD.md), with the model name, variant, and checkpoint step filled in by the exporter.

## Upload a New Model Bundle

Sign in as `larry077` with a Hugging Face token that has write access:

```bash
hf auth login
hf auth whoami

python tools/upload_weights.py \
  --bundle /path/to/ReentryMem-model \
  --repo-id larry077/ReentryMem --dry-run

python tools/upload_weights.py \
  --bundle /path/to/ReentryMem-model \
  --repo-id larry077/ReentryMem --public
```

The uploader retries transient network errors and checks remote commit state before repeating a transfer. It makes a staged repository public only after file sizes and hashes match. Re-running against an identical public bundle succeeds without another upload. An existing public repository with different files is not overwritten by this tool.

For model-card edits, update `README.md` and its entry in `checksums.json` together in one Hugging Face commit. Use `HfApi.create_commit` with only those two files; the transformer file does not need to be transferred again.

## Update the Source Repository

After a web edit to the GitHub README, synchronize it before making local changes:

```bash
git pull --ff-only
```

Commit the intended source changes and push to `main`. Keep model weights, generated videos, and credentials out of source commits. The original project video is explicitly allowed by `.gitignore`.

`tools/publish_github.py` is for the initial creation of a repository. It uses `GH_TOKEN`, `GITHUB_TOKEN`, or an existing GitHub CLI login, and expects an unused repository name. For the existing `Larry-077/ReentryMem` repository, use `git push`.

## Source Layout

```text
infer.py                    Inference entry point
reentrymem/pipeline.py       Video continuation and denoising
reentrymem/regions.py        Entity-attention readout and region extraction
reentrymem/sampling.py       Localized sampling correction
reentrymem/modules/          Transformer, text encoder, and VAE
tools/                      Input preparation and release utilities
tests/                      CPU and upload-recovery checks
```
