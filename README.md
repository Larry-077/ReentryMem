# ReentryMem

**Out of View, Still in Motion: Dynamic Entity Re-entry in Video World Models**

ReentryMem improves entity re-entry in camera-controlled video generation. It reads entity regions from text cross-attention and uses them to localize sampling correction. This repository contains inference code, calibrated region settings, and checkpoint export tools.

## Setup

Use Python 3.10 or later with a CUDA-enabled PyTorch installation.

```bash
conda create -n reentrymem python=3.11 -y
conda activate reentrymem
pip install -r requirements.txt
```

FlashAttention is optional. Without it, CUDA inference uses fused PyTorch attention. CPU execution is provided for tests and input validation.

## Models

The released checkpoint contains the camera-conditioned diffusion transformer. The tokenizer, text encoder, and VAE come from the Wan2.2 base model:

```bash
python tools/download_models.py
```

Once the ReentryMem weights are uploaded to Hugging Face, download a bundle with its actual repository ID:

```bash
python tools/download_models.py \
  --repo-id YOUR_HF_ACCOUNT/ReentryMem \
  --checkpoint-dir checkpoints/ReentryMem
```

You can also export an existing checkpoint as described below. The source repository does not contain model weights.

## Inference

Each input contains a 77-frame observed video at 15 fps, a caption, entity phrases, and planned camera controls. Generation produces 76 future frames at 832×480. Entity phrases must occur in the caption. No future RGB frames or target masks are supplied to the model.

Create a case from your observed video:

```bash
python tools/create_case.py \
  --video /path/to/observed.mp4 \
  --prompt "A girl walks through a stylized town." \
  --subjects "girl" \
  --preset turn-return \
  --output examples/case.json
```

For existing planned camera poses, use `--camera-json camera.json` in place of `--preset`. Alternatively, `--actions-json actions.json` accepts 39 integer labels: 20 observed and 19 generated latent frames. The label is `translation * 9 + rotation`; each component has nine actions, including stationary.

```bash
python infer.py \
  --case examples/case.json \
  --checkpoint checkpoints/ReentryMem \
  --base-model pretrained/Wan2.2-TI2V-5B \
  --output-dir outputs/demo
```

Defaults match the main inference setup: 50 steps, CFG 5, sigma shift 5, generation seed 42, correction seed 0, correction strength 8, and noise gate 0.92. `--sampler euler` uses the same weights with ordinary sampling. `--save-masks` additionally saves the entity-region fields.

Outputs are `video.mp4` with the observed prefix and continuation, `continuation.mp4` with the generated segment, and `receipt.json` with the settings and runtime measurements.

```bash
python infer.py \
  --case examples/case.json \
  --checkpoint checkpoints/ReentryMem \
  --base-model pretrained/Wan2.2-TI2V-5B \
  --dry-run --verify-checkpoint
```

Multiple entity phrases are supported through the union of their regions. Use the multi-entity checkpoint for the corresponding trained variant. A block-causal checkpoint reads its attention mode from `config.json` and retains joint target denoising.

## Checkpoint export and Hugging Face

Export a model folder without optimizer state or training data:

```bash
python tools/export_checkpoint.py \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/ReentryMem-model \
  --name ReentryMem --variant single-entity --step 1496
```

The bundle contains the merged transformer, architecture configuration, calibrated regions, inference defaults, model card, license, and SHA256 checksums. `--hardlink` can stage the weights locally without an additional copy; exported linked weights must remain unchanged.

After signing in with `hf auth login`, upload the prepared bundle:

```bash
python tools/upload_weights.py \
  --bundle /path/to/ReentryMem-model \
  --repo-id YOUR_HF_ACCOUNT/ReentryMem --public
```

The upload starts in a private model repository and makes it public after the expected files are present. Weights and source code are published separately.

## Source layout

```text
infer.py                    Inference entry point
reentrymem/pipeline.py       Video continuation and denoising
reentrymem/regions.py        Entity attention and region extraction
reentrymem/sampling.py       Localized sampling correction
reentrymem/modules/          Backbone, text encoder, and VAE
tools/                      Input preparation and model publishing
tests/                      CPU contract and numerical checks
```

```bash
python -m unittest discover -s tests -v
```

To create the GitHub repository after the initial local commit, authenticate with `gh auth login` or `GH_TOKEN`, then run `python tools/publish_github.py`. The default repository name is `ReentryMem`.

## Acknowledgments

The backbone and camera-conditioning code build on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [BiWM](https://github.com/LynnReal-AI/BiWM). Upstream attribution and sampling provenance are preserved in [NOTICE](NOTICE). The code is distributed under Apache-2.0.
