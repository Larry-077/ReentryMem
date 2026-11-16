import hashlib
import json
from pathlib import Path

MODEL_NAME = "diffusion_pytorch_model.safetensors"
MODEL_DEFAULTS = dict(
    model_type="ti2v",
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    in_dim=48,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    patch_size=[1, 2, 2],
    text_len=512,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-06,
)
CONFIG_KEYS = set(MODEL_DEFAULTS) | {"temporal_attention_mode", "temporal_attention_backend"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size <= 64 << 20:
            raise ValueError("Invalid safetensors header size")
        data = json.loads(stream.read(size))
    return {k: v for k, v in data.items() if k != "__metadata__"}


def preflight(checkpoint, base_model, verify=False):
    checkpoint, base_model = (Path(checkpoint).resolve(), Path(base_model).resolve())
    needed = [
        checkpoint / MODEL_NAME,
        checkpoint / "config.json",
        base_model / "Wan2.2_VAE.pth",
        base_model / "models_t5_umt5-xxl-enc-bf16.pth",
        base_model / "google/umt5-xxl/tokenizer_config.json",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing model assets:\n" + "\n".join(missing))
    config = json.loads((checkpoint / "config.json").read_text())
    unknown = set(config) - CONFIG_KEYS - {k for k in config if k.startswith("_")}
    if unknown:
        raise ValueError(f"Unsupported model config entries: {sorted(unknown)}")
    config = {**MODEL_DEFAULTS, **{k: v for k, v in config.items() if k in CONFIG_KEYS}}
    if (config["dim"], config["num_heads"], config["num_layers"], config["in_dim"]) != (
        3072,
        24,
        30,
        48,
    ):
        raise ValueError("This release supports the camera-conditioned Wan2.2-TI2V-5B checkpoint")
    header = safetensors_header(checkpoint / MODEL_NAME)
    if "hycam_action_embedder.mlp.0.weight" not in header:
        raise ValueError("Checkpoint does not contain the camera-action parameters")
    if verify:
        verify_bundle(checkpoint)
    return dict(
        config=config, tensors=len(header), weight_bytes=(checkpoint / MODEL_NAME).stat().st_size
    )


def verify_bundle(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "checksums.json").read_text())
    for name, expected in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Checksum path escapes the model bundle")
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Checkpoint verification failed: {name}")
    return manifest
