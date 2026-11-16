import numpy as np

VAE_STRIDE = 4
NUM_LATENT = 39
_TRANS = {
    (0, 0): 0,
    (1, 0): 1,
    (-1, 0): 2,
    (0, 1): 3,
    (0, -1): 4,
    (1, 1): 5,
    (1, -1): 6,
    (-1, 1): 7,
    (-1, -1): 8,
}
_ROT = {
    (0, 0): 0,
    (1, 0): 1,
    (-1, 0): 2,
    (0, 1): 3,
    (0, -1): 4,
    (1, 1): 5,
    (1, -1): 6,
    (-1, 1): 7,
    (-1, -1): 8,
}


def _sign(v, thr):
    return int(v > thr) - int(v < -thr)


def poses_to_action_labels(poses, trans_thr_frac=0.05, yaw_thr_deg=0.5, pitch_thr_deg=0.5):
    latent_poses = poses[::VAE_STRIDE]
    inv = np.linalg.inv(latent_poses[:-1])
    rel = np.einsum("nij,njk->nik", inv, latent_poses[1:])
    t = rel[:, :3, 3]
    norms = np.linalg.norm(t, axis=1)
    moving = norms[norms > 1e-06]
    trans_thr = max(1.0, trans_thr_frac * float(np.median(moving))) if moving.size else 1.0
    R = rel[:, :3, :3]
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    pitch = np.degrees(-np.arcsin(np.clip(R[:, 2, 0], -1.0, 1.0)))
    labels = [0]
    for i in range(len(rel)):
        tr = _TRANS[_sign(t[i, 0], trans_thr), _sign(t[i, 1], trans_thr)]
        ro = _ROT[_sign(pitch[i], pitch_thr_deg), _sign(yaw[i], yaw_thr_deg)]
        labels.append(tr * 9 + ro)
    return labels[:NUM_LATENT]


def load_camera_actions(path):
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    poses = np.asarray(
        [data["cond_cam"][str(i)] for i in range(77)]
        + [data["tgt_cam"][str(i)] for i in range(76)],
        dtype=np.float64,
    )
    if poses.shape != (153, 4, 4):
        raise ValueError("Expected 153 planned camera poses")
    return [int(v) for v in poses_to_action_labels(poses)]
