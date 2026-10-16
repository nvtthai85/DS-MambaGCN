"""NumPy-only data utilities (no PyTorch import, so they can be unit-tested anywhere).

Array layout throughout: (C, T, V, M) per sample, C = 3 coordinates.
"""
import numpy as np

from .graph import SPINE_BASE


def valid_length(seq: np.ndarray) -> int:
    """Number of leading frames that contain a non-zero skeleton (NTU zero-pads to 300)."""
    nonzero = np.abs(seq).sum(axis=(0, 2, 3)) > 0
    idx = np.flatnonzero(nonzero)
    return int(idx[-1] + 1) if idx.size else 0


def uniform_resample(seq: np.ndarray, length: int, T_out: int) -> np.ndarray:
    """Linearly resample the first `length` frames of `seq` to exactly T_out frames."""
    C, _, V, M = seq.shape
    if length <= 1:
        return np.repeat(seq[:, :1], T_out, axis=1)
    src = np.linspace(0, length - 1, T_out)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, length - 1)
    w = (src - lo)[None, :, None, None]
    return (1 - w) * seq[:, lo] + w * seq[:, hi]


def center_on_spine(seq: np.ndarray) -> np.ndarray:
    """Subtract the spine-base joint of body 0 in the first frame from every joint."""
    origin = seq[:, 0, SPINE_BASE, 0].copy()
    mask = (np.abs(seq).sum(axis=0, keepdims=True) > 0)      # keep zero-padded bodies at zero
    return (seq - origin[:, None, None, None]) * mask


def random_rotation(seq: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, size=3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    R = rz @ ry @ rx
    return np.einsum("ij,jtvm->itvm", R, seq)


def temporal_shift(seq: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Shift by up to +/- max_shift frames, padding with the edge frame."""
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return seq
    if s > 0:
        return np.concatenate([np.repeat(seq[:, :1], s, axis=1), seq[:, :-s]], axis=1)
    return np.concatenate([seq[:, -s:], np.repeat(seq[:, -1:], -s, axis=1)], axis=1)


def augment(seq: np.ndarray, rng: np.random.Generator, rot_deg=15.0, shift=5,
            noise_std=0.01) -> np.ndarray:
    """Augmentation of Table 7: rotation +/-15 deg, shift +/-5 frames, Gaussian noise 0.01."""
    out = seq
    if rot_deg:
        out = random_rotation(out, rot_deg, rng)
    if shift:
        out = temporal_shift(out, shift, rng)
    if noise_std:
        out = out + rng.normal(0.0, noise_std, size=out.shape)
    return out


def stratified_holdout(labels: np.ndarray, frac: float, seed: int):
    """Split indices into (train, val) keeping class proportions; deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        k = max(1, int(round(frac * idx.size)))
        val.extend(idx[:k].tolist())
    val = np.sort(np.array(val, dtype=int))
    train = np.setdiff1d(np.arange(labels.size), val)
    return train, val


def load_ntu_npz(path: str):
    """Load the CTR-GCN/InfoGCN NTU file (e.g. NTU60_CS.npz).

    x_*: (N, 300, 150) = 2 bodies x 25 joints x 3 coords; y_*: one-hot (N, K).
    Returns dict with x_train/x_test as (N, 3, 300, 25, 2) and integer labels.
    """
    d = np.load(path)
    out = {}
    for split in ("train", "test"):
        x = d[f"x_{split}"]
        N, T, _ = x.shape
        x = x.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)
        y = d[f"y_{split}"]
        out[f"x_{split}"] = x.astype(np.float32)
        out[f"y_{split}"] = (y.argmax(-1) if y.ndim == 2 else y).astype(np.int64)
    return out


# Official training subjects / cameras / setups (Shahroudy et al. 2016; Liu et al. 2020).
NTU60_XSUB_TRAIN = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
NTU60_XVIEW_TRAIN_CAMERAS = {2, 3}
NTU120_XSUB_TRAIN = NTU60_XSUB_TRAIN | {
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85,
    86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103}


def ntu_split_mask(meta: dict, benchmark: str, dataset: str = None):
    """Boolean train mask for benchmark in {xsub, xview, xset}; test = ~mask.

    dataset ("ntu60" / "ntu120") selects the X-Sub subject list; if omitted it is
    inferred from the labels present."""
    if benchmark == "xsub":
        if dataset is None:
            dataset = "ntu120" if int(meta["label"].max()) >= 60 else "ntu60"
        train_ids = NTU120_XSUB_TRAIN if dataset == "ntu120" else NTU60_XSUB_TRAIN
        return np.isin(meta["subject"], list(train_ids))
    if benchmark == "xview":
        return np.isin(meta["camera"], list(NTU60_XVIEW_TRAIN_CAMERAS))
    if benchmark == "xset":
        return meta["setup"] % 2 == 0
    raise ValueError(benchmark)


class CompactSkeletons:
    """Variable-length samples stored back to back (tools/ntu_preprocess.py).

    Indexing returns one sample as (C=3, T_i, V=25, M=2) float32, ready for
    uniform_resample. The frame array is memory-mapped, so DataLoader workers
    share it without copying.
    """

    def __init__(self, root: str, indices=None):
        self.root = root
        self.data = np.load(f"{root}/data.npy", mmap_mode="r")
        m = np.load(f"{root}/meta.npz")
        self.meta = {k: m[k] for k in m.files}
        self.indices = np.arange(len(self.meta["label"])) if indices is None else np.asarray(indices)

    def subset(self, indices):
        out = CompactSkeletons.__new__(CompactSkeletons)
        out.root, out.data, out.meta = self.root, self.data, self.meta
        out.indices = self.indices[np.asarray(indices)]
        return out

    @property
    def labels(self):
        return self.meta["label"][self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        k = self.indices[i]
        o, n = int(self.meta["offset"][k]), int(self.meta["length"][k])
        return np.asarray(self.data[o:o + n]).transpose(3, 0, 2, 1)  # (3, T, 25, 2)


def read_name_list(path: str):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
