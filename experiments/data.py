"""Original-coordinate patches; target training datasets never store class labels."""
from pathlib import Path
import hashlib
import random
import numpy as np
import scipy.io
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Grass healthy', 'Grass stressed', 'Trees', 'Water',
               'Residential buildings', 'Non-residential buildings', 'Road']
EXPECTED = {'Houston13': [345,365,365,285,319,408,443],
            'Houston18': [1353,4888,2766,22,5347,32459,6365]}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_mat(path, key):
    try:
        return scipy.io.loadmat(path)[key]
    except (NotImplementedError, ValueError):
        with h5py.File(path) as f:
            return np.asarray(f[key]).T


def load_scene(root, name):
    root = Path(root)
    image_path, label_path = root / f'{name}.mat', root / f'{name}_7gt.mat'
    image = read_mat(image_path, 'ori_data').astype(np.float32)
    labels = read_mat(label_path, 'map').astype(np.int64) - 1
    if image.shape[:2] != labels.shape or image.shape[-1] != 48 or not np.isfinite(image).all():
        raise ValueError(f'Invalid Houston data: {name} {image.shape} {labels.shape}')
    support = np.bincount(labels[labels >= 0], minlength=7).tolist()
    if support != EXPECTED[name]:
        raise ValueError(f'{name}: expected paper class counts {EXPECTED[name]}, got {support}')
    # Same per-pixel L2 spectral normalization for all models. MLUDA adds ILDA.
    image /= np.maximum(np.linalg.norm(image, axis=-1, keepdims=True), 1e-12)
    return image, labels, {str(p.name): digest(p) for p in (image_path, label_path)}


def source_split(labels, seed, ratio=.95):
    coords = np.argwhere(labels >= 0)
    train, val = train_test_split(np.arange(len(coords)), train_size=ratio,
        stratify=labels[tuple(coords.T)], random_state=seed)
    return coords[train], coords[val]


class Patches(Dataset):
    def __init__(self, image, coordinates, patch_size, labels=None):
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        self.patch_size = patch_size
        before = patch_size // 2
        after = patch_size - 1 - before
        self.image = np.pad(image, ((before, after), (before, after), (0, 0)), mode='reflect')
        self.labels = None if labels is None else labels[tuple(self.coordinates.T)].copy()

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, i):
        r, c = self.coordinates[i]
        p = self.image[r:r+self.patch_size, c:c+self.patch_size].transpose(2, 0, 1).copy()
        x = torch.from_numpy(p)
        return (x, int(self.labels[i])) if self.labels is not None else (x, i)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def augment(x, sigma=.05):
    """Independent per-sample rotations/reflections and Gaussian noise; center is preserved."""
    out = x.clone()
    for k in range(4):
        # Disjoint random groups, not repeated transforms of a changing mask.
        if k == 0:
            rotation = torch.randint(4, (len(x),), device=x.device)
        mask = rotation == k
        out[mask] = torch.rot90(x[mask], k, (-2, -1))
    mask = torch.rand(len(x), device=x.device) < .5
    out[mask] = out[mask].flip(-1)
    return out + sigma * torch.randn_like(out)
