#!/usr/bin/env python3
"""Create a portable case from an observed video and planned camera controls."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-id")
    controls = p.add_mutually_exclusive_group(required=True)
    controls.add_argument("--camera-json", help="Planned cond_cam/tgt_cam camera-to-world poses")
    controls.add_argument("--actions-json", help="JSON list of 39 integer labels")
    controls.add_argument("--preset", choices=["stationary", "turn-return"])
    a = p.parse_args()
    if a.camera_json:
        from reentrymem.camera import load_camera_actions

        actions = load_camera_actions(a.camera_json)
    elif a.actions_json:
        actions = json.loads(Path(a.actions_json).read_text())
    else:
        actions = [0] * 39 if a.preset == "stationary" else [0] * 20 + [3] * 9 + [4] * 10
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        sample_id=a.sample_id or Path(a.video).stem,
        condition_video=os.path.relpath(Path(a.video).resolve(), output.parent),
        prompt=a.prompt,
        subjects=a.subjects,
        actions=actions,
    )
    output.write_text(json.dumps(data, indent=2) + "\n")
    from reentrymem.case import load_case

    load_case(output)
    print(output)


if __name__ == "__main__":
    main()
