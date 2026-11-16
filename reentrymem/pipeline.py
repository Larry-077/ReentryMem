from contextlib import ExitStack, contextmanager
import gc
import json
from pathlib import Path
import time
import torch
from .case import stable_seed, subject_indices
from .constants import WAN_NEG_PROMPT_TEXT
from .runtime import VAE, load_dit, load_text_encoder
from .regions import EntityAttentionReader, load_region_config, region_head_sets
from .sampling import ReentryMemSampler, EntityRegions
from .video import read_condition, write_video


def shifted_sigmas(steps, shift, device):
    if steps <= 0 or shift <= 0:
        raise ValueError("steps and shift must be positive")
    sigma = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return shift * sigma / (1 + (shift - 1) * sigma)


@torch.no_grad()
def sample(
    model,
    initial,
    prefix,
    positive,
    negative,
    actions,
    *,
    steps=50,
    cfg=5.0,
    shift=5.0,
    sampler=None,
    read_regions=None,
    progress=None,
):
    device = initial.device
    latent = initial.float()
    count = prefix.shape[1]
    tokens = latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)
    sigmas = shifted_sigmas(steps, shift, device)
    for step in range(steps):
        sigma, following = (float(sigmas[step]), float(sigmas[step + 1]))
        x = latent.float()
        inputs = [x[0].to(torch.bfloat16)]
        timestep = torch.tensor([sigma * 1000], device=device)
        with (
            read_regions(step, sigma)
            if read_regions
            else torch.autocast("cuda", dtype=torch.bfloat16)
        ):
            pos = torch.stack(
                model(
                    x=inputs,
                    t=timestep,
                    context=[positive],
                    seq_len=tokens,
                    cond_latent_frames=count,
                    cond_timestep="zero",
                    action_labels=actions,
                )
            )
        if cfg > 1:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                neg = torch.stack(
                    model(
                        x=inputs,
                        t=timestep,
                        context=[negative],
                        seq_len=tokens,
                        cond_latent_frames=count,
                        cond_timestep="zero",
                    )
                )
            velocity = neg + cfg * (pos - neg)
        else:
            velocity = pos
        x0 = x - sigma * velocity.float()
        x0[0, :, :count] = prefix
        latent = (
            x + velocity.float() * (following - sigma)
            if sampler is None
            else sampler.step(step, sigma, following, x, velocity.float(), x0)
        )
        latent[0, :, :count] = prefix
        if progress:
            progress(step + 1, steps)
    return latent


class UnionRegions:
    def __init__(self, sources):
        self.sources = sources

    def gain(self, step, sigma, sample, x0_hat):
        return torch.stack([s.gain(step, sigma, sample, x0_hat) for s in self.sources]).amax(0)


@torch.no_grad()
def generate(
    case,
    checkpoint,
    base_model,
    output_dir,
    *,
    region_spec,
    sampler_name="localized",
    steps=50,
    cfg=5.0,
    shift=5.0,
    seed=42,
    correction_seed=0,
    c=8.0,
    sigma_max=0.92,
    device="cuda:0",
    save_masks=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires an available CUDA GPU")
    if sampler_name not in ["localized", "euler"]:
        raise ValueError("sampler must be localized or euler")
    if sampler_name == "localized" and (not case.subjects):
        raise ValueError("Full requires at least one subject phrase in the prompt")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    pixels = read_condition(case.condition_video)
    model, checkpoint_report = load_dit(checkpoint, base_model, device)
    encoder = load_text_encoder(base_model, device)

    def encode(prompts, text_encoder=encoder):
        return text_encoder(prompts, device)

    print("Encoding camera controls and prompts", flush=True)
    model.precompute_cam_text_embeddings(encode, device, torch.bfloat16)
    positive = encode([case.prompt])[0].to(device=device, dtype=torch.bfloat16)
    negative = encode([case.negative_prompt or WAN_NEG_PROMPT_TEXT])[0].to(
        device=device, dtype=torch.bfloat16
    )
    token_sets = []
    if sampler_name == "localized":
        if case.subject_token_indices is not None:
            token_sets = [list(v) for v in case.subject_token_indices]
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(Path(base_model) / "google/umt5-xxl"), use_fast=True, local_files_only=True
            )
            token_sets = [
                subject_indices(case.prompt, phrase, tokenizer) for phrase in case.subjects
            ]
    del encoder, encode
    gc.collect()
    torch.cuda.empty_cache()
    vae = VAE(base_model, device)
    prefix = vae.encode(
        torch.from_numpy(pixels.copy()).float().permute(3, 0, 1, 2).to(device) / 127.5 - 1
    )
    if tuple(prefix.shape) != (48, 20, 30, 52):
        raise RuntimeError(f"Unexpected observed latent shape: {tuple(prefix.shape)}")
    rng_seed = stable_seed(case.sample_id, seed)
    rng = torch.Generator(device=device).manual_seed(rng_seed)
    initial = torch.randn((1, 48, 39, 30, 52), generator=rng, device=device, dtype=torch.float32)
    initial[0, :, :20] = prefix
    actions = torch.tensor([case.actions], device=device, dtype=torch.long)
    sampler = read_regions = None
    if sampler_name == "localized":
        spec = load_region_config(region_spec, blocks=len(model.blocks), heads=model.num_heads)
        sources = [
            EntityRegions(
                spec, frames=39, height=15, width=26, cond_frames=20, trusted_sigma=sigma_max
            )
            for _ in token_sets
        ]
        sampler = ReentryMemSampler(
            c,
            UnionRegions(sources),
            cond_latent_frames=20,
            seed=stable_seed(case.sample_id, correction_seed),
            device=device,
            keep_fields=save_masks,
        )

        @contextmanager
        def read_regions(step, sigma):
            readers = [
                EntityAttentionReader(
                    model,
                    indices,
                    frames=39,
                    tokens_per_frame=390,
                    head_sets=region_head_sets(spec),
                )
                for indices in token_sets
            ]
            with ExitStack() as stack:
                stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
                for reader in readers:
                    stack.enter_context(reader)
                yield
            for source, reader in zip(sources, readers):
                source.capture_from(reader)

    print(f"Generating {sampler_name}, {steps} steps, seed={seed}", flush=True)
    latent = sample(
        model,
        initial,
        prefix,
        positive,
        negative,
        actions,
        steps=steps,
        cfg=cfg,
        shift=shift,
        sampler=sampler,
        read_regions=read_regions,
        progress=lambda n, total: print(f"step {n}/{total}", flush=True),
    )
    frames = vae.decode(latent[0])
    if frames.shape != (153, 480, 832, 3):
        raise RuntimeError(f"Unexpected decoded video shape: {frames.shape}")
    write_video(output_dir / "video.mp4", frames)
    write_video(output_dir / "continuation.mp4", frames[77:])
    if save_masks and sampler is not None:
        import numpy as np

        np.savez_compressed(
            output_dir / "entity_gains.npz",
            gain=torch.stack(sampler.fields).numpy(),
            sigma=np.array([r["sigma"] for r in sampler.trace]),
            c=c,
        )
    receipt = dict(
        sample_id=case.sample_id,
        sampler=sampler_name,
        prompt=case.prompt,
        subjects=list(case.subjects),
        subject_token_indices=token_sets,
        resolution=[832, 480],
        fps=15,
        observed_frames=77,
        generated_frames=76,
        steps=steps,
        cfg=cfg,
        sigma_shift=shift,
        seed=seed,
        derived_seed=rng_seed,
        correction_seed=correction_seed,
        correction_strength=c if sampler_name == "localized" else 0,
        sigma_max=sigma_max,
        temporal_attention=model.temporal_attention_mode,
        weight_bytes=checkpoint_report["weight_bytes"],
        model_config=checkpoint_report["config"],
        wall_seconds=time.monotonic() - start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 1024**3,
        output="video.mp4",
        continuation="continuation.mp4",
    )
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
