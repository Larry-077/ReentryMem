from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def stable_seed(sample_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class Case:
    sample_id: str
    condition_video: Path
    prompt: str
    subjects: tuple[str, ...]
    actions: tuple[int, ...]
    subject_token_indices: tuple[tuple[int, ...], ...] | None = None
    negative_prompt: str | None = None


def load_case(path: str | Path) -> Case:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    video = Path(data["condition_video"]).expanduser()
    if not video.is_absolute():
        video = path.parent / video
    if not video.is_file():
        raise FileNotFoundError(f"Condition video does not exist: {video}")
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("prompt must be a nonempty string")
    subjects = data.get("subjects", [])
    if not isinstance(subjects, list) or any(
        (not isinstance(s, str) or not s.strip() for s in subjects)
    ):
        raise ValueError("subjects must be a list of nonempty phrases from the prompt")
    subjects = tuple((s.strip() for s in subjects))
    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) != 39:
        raise ValueError("actions must contain 39 latent-frame camera labels")
    if any((isinstance(v, bool) or not isinstance(v, int) or (not 0 <= v <= 80) for v in actions)):
        raise ValueError("Every camera action must be an integer in [0, 80]")
    indices = data.get("subject_token_indices")
    if indices is not None:
        if len(indices) != len(subjects) or any(
            (
                not row
                or any((isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in row))
                for row in indices
            )
        ):
            raise ValueError(
                "subject_token_indices must contain one nonempty index list per subject"
            )
        indices = tuple((tuple(row) for row in indices))
    return Case(
        str(data.get("sample_id", video.stem)),
        video.resolve(),
        prompt,
        subjects,
        tuple(actions),
        indices,
        data.get("negative_prompt"),
    )


def subject_indices(prompt, phrase, tokenizer):
    import re

    text = re.sub("\\s+", " ", prompt).strip()
    spans = [
        (m.start(), m.end())
        for m in re.finditer("(?<!\\w)" + re.escape(phrase) + "(?!\\w)", text, re.IGNORECASE)
    ]
    if not spans:
        raise ValueError(f"Subject phrase {phrase!r} is absent from the prompt")
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    result = sorted(
        {
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            for lo, hi in spans
            if b > a and a < hi and (b > lo)
        }
    )
    if not result:
        raise ValueError(f"No text tokens found for subject {phrase!r}")
    return result
