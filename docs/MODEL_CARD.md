---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
base_model_relation: finetune
library_name: pytorch
language:
- en
thumbnail: https://reentrymem-project.pages.dev/media/reentrymem-final.jpg
tags:
- video-generation
- video-to-video
- world-model
- camera-control
- entity-reentry
- reentrymem
---

<h1 align="center">${model_name}</h1>

<p align="center">
  <strong>Out of View, Still in Motion:<br>Dynamic Entity Re-entry in Video World Models</strong>
</p>

<p align="center">
  <a href="https://reentrymem-project.pages.dev/">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="https://github.com/Larry-077/ReentryMem">💻 Code</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B">Base Model</a>
</p>

ReentryMem addresses **dynamic entity re-entry** in camera-controlled video world models: recovering a moving subject after it leaves the field of view and later returns. It reads entity presence and spatial regions from text cross-attention on noisy video latents. The shared readout guides entity-weighted flow-matching training and localized overshooting with re-noising during inference.

<video controls playsinline preload="none" width="100%" poster="https://reentrymem-project.pages.dev/media/reentrymem-final.jpg" src="https://reentrymem-project.pages.dev/media/reentrymem-final.mp4"></video>

Explore the [project page](https://reentrymem-project.pages.dev/) for synchronized video comparisons, the method, and more qualitative results.

## Model Overview

| Property | Specification |
|---|---|
| Released variant | ${variant} |
| Base model | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Parameters | 5.01B |
| Weights | Complete merged DiT with camera-conditioning parameters |
| Precision | BF16 |
| Training checkpoint | Step ${checkpoint_step} |
| Inputs | Observed video, caption, subject phrase, and planned camera controls |
| Continuation | 76 frames at 832×480, 15 fps |

The single-entity checkpoint was fine-tuned using entity-weighted flow matching with internally predicted entity regions. The released transformer is accompanied by the calibrated region settings and inference defaults. Its tokenizer, UMT5 text encoder, and VAE are obtained separately from Wan2.2.

## How ReentryMem Works

![ReentryMem method overview](https://reentrymem-project.pages.dev/media/figure-method.png)

- **Localize during denoising.** Selected cross-attention heads estimate entity presence and shape directly from noisy latents.
- **Emphasize entity supervision.** The predicted regions weight the flow-matching objective during training.
- **Correct the entity region.** At inference, localized overshooting and re-noising use the same region readout; other positions retain the Euler update.

`localized` and `euler` are two sampling choices for this **same checkpoint**. The former is the full ReentryMem inference configuration; the latter corresponds to ReentryMem-Train in the controlled comparison.

## Getting Started

Use the [ReentryMem inference repository](https://github.com/Larry-077/ReentryMem) with Python 3.10 or later and CUDA-enabled PyTorch:

```bash
git clone https://github.com/Larry-077/ReentryMem.git
cd ReentryMem
conda create -n reentrymem python=3.11 -y
conda activate reentrymem
pip install -r requirements.txt

python tools/download_models.py \
  --repo-id larry077/ReentryMem \
  --checkpoint-dir checkpoints/ReentryMem
```

Prepare a 77-frame observed video at 15 fps. The subject phrase must appear in the caption:

```bash
python tools/create_case.py \
  --video /path/to/observed.mp4 \
  --prompt "A girl walks through a stylized town." \
  --subjects "girl" \
  --preset turn-return \
  --output examples/case.json

python infer.py \
  --case examples/case.json \
  --checkpoint checkpoints/ReentryMem \
  --base-model pretrained/Wan2.2-TI2V-5B \
  --output-dir outputs/demo
```

The default uses localized sampling, 50 steps, CFG 5, sigma shift 5, and correction strength 8. Add `--sampler euler` for ordinary sampling or `--save-masks` to save entity-region fields. The output includes the complete video, the generated continuation, and a settings receipt. See the [code documentation](https://github.com/Larry-077/ReentryMem) for camera inputs and additional options.

## Evaluation

The controlled comparison reported on the [project page](https://reentrymem-project.pages.dev/#comparisons) is:

| Method | Rea-Score (%) ↑ | SC ↑ | TCM ↑ |
|---|---:|---:|---:|
| Baseline | 96.1 | 0.8204 | 0.9154 |
| ReentryMem-Train | 99.2 | 0.8380 | 0.9231 |
| **ReentryMem** | 97.2 | 0.8378 | 0.9318 |

The protocol separately reports entity reappearance, subject consistency, and temporal consistency match. Refer to the project page for the full comparisons and qualitative examples.

## Usage Notes

This release targets single-entity re-entry with the stated observed/future clip lengths and camera controls. Use the custom ReentryMem inference implementation to load the camera-conditioned transformer. GPU memory use depends on the attention backend and runtime settings.

The research results above come from the project's evaluation runs. The standalone package has passed CPU tests, strict full-weight loading, and a small forward check with synthetic text features; full-resolution CUDA generation has not yet been re-run for this package. Validation details accompany the [code release](https://github.com/Larry-077/ReentryMem).

## License and Acknowledgments

The released weights and inference code use [Apache-2.0](https://huggingface.co/larry077/ReentryMem/blob/main/LICENSE). ReentryMem builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [BiWM](https://github.com/LynnReal-AI/BiWM). See [NOTICE](https://huggingface.co/larry077/ReentryMem/blob/main/NOTICE) for upstream attribution and sampling provenance.
