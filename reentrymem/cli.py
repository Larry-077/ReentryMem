import argparse
import json
from importlib.resources import files
from pathlib import Path
from .case import load_case
from .checkpoints import preflight


def main():
    parser = argparse.ArgumentParser(description="ReentryMem: observed-video continuation")
    parser.add_argument("--case", required=True, help="JSON input manifest")
    parser.add_argument(
        "--checkpoint", required=True, help="ReentryMem checkpoint bundle directory"
    )
    parser.add_argument(
        "--base-model", required=True, help="Wan2.2 text encoder, tokenizer and VAE directory"
    )
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument(
        "--sampler",
        choices=["localized", "euler"],
        default="localized",
        help="localized correction or standard Euler sampling",
    )
    parser.add_argument("--region-spec", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correction-seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=8.0)
    parser.add_argument("--sigma-max", type=float, default=0.92)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument(
        "--verify-checkpoint", action="store_true", help="Verify the exported SHA256 manifest"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate files and configuration without a GPU"
    )
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.shift <= 0
        or args.cfg < 0
        or (args.c < 0)
        or (not 0 < args.sigma_max <= 1)
    ):
        parser.error("Invalid sampling parameters")
    case = load_case(args.case)
    report = preflight(args.checkpoint, args.base_model, args.verify_checkpoint)
    if args.region_spec is None:
        bundled = Path(args.checkpoint) / "entity_regions.json"
        args.region_spec = str(
            bundled
            if bundled.is_file()
            else files("reentrymem").joinpath("configs/entity_regions.json")
        )
    if args.sampler == "localized" and (not case.subjects):
        parser.error("--sampler localized requires subjects in the case manifest")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    sample_id=case.sample_id,
                    sampler_name=args.sampler,
                    checkpoint=report,
                    resolution=[832, 480],
                    observed_frames=77,
                    generated_frames=76,
                ),
                indent=2,
            )
        )
        return
    from .pipeline import generate

    result = generate(
        case,
        args.checkpoint,
        args.base_model,
        args.output_dir,
        region_spec=args.region_spec,
        sampler_name=args.sampler,
        steps=args.steps,
        cfg=args.cfg,
        shift=args.shift,
        seed=args.seed,
        correction_seed=args.correction_seed,
        c=args.c,
        sigma_max=args.sigma_max,
        device=args.device,
        save_masks=args.save_masks,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
