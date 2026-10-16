"""NTU RGB+D 25-joint skeleton (Kinect v2) and its normalised adjacency (Eq. 5)."""
import numpy as np

NUM_JOINTS = 25

# 1-based (child, parent) pairs, the 24 bones of the Kinect v2 skeleton
# (same list as ST-GCN / CTR-GCN `ntu_rgb_d.py`).
BONES_1BASED = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
    (24, 25), (25, 12),
]
BONES = [(i - 1, j - 1) for i, j in BONES_1BASED]

# 0-based index of the spine-base joint used for body-centring.
SPINE_BASE = 0


def physical_adjacency(num_joints: int = NUM_JOINTS) -> np.ndarray:
    """Binary symmetric adjacency A of Eq. (4), without self-loops."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in BONES:
        A[i, j] = A[j, i] = 1.0
    return A


def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} (A + I) D^{-1/2} of Eq. (5)."""
    A_hat = A + np.eye(A.shape[0], dtype=A.dtype)
    deg = A_hat.sum(1)
    d_inv_sqrt = np.power(deg, -0.5)
    return (d_inv_sqrt[:, None] * A_hat * d_inv_sqrt[None, :]).astype(np.float32)


assert len(BONES) == 24
