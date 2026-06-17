"""Dataset classes and DataLoader factories for the AstroAI tutorial."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class LCDataset(Dataset):
    """ZTF interpolated light curve dataset.

    Each sample x has shape (T, 4): [g-flux, r-flux, g-mask, r-mask].
    Flux channels are normalized by training-set statistics; unobserved
    steps (mask=0) are reset to 0 after normalization.
    """
    def __init__(self, fluxes: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                 flux_mean: np.ndarray | None = None, flux_std: np.ndarray | None = None):
        fluxes = fluxes.copy()
        if flux_mean is not None:
            fluxes = (fluxes - flux_mean) / (flux_std + 1e-8)
            fluxes[masks == 0] = 0.0  # reset unobserved to neutral after normalization
        x = np.concatenate([fluxes, masks], axis=-1).astype(np.float32)  # (N, T, 4)
        self.x = torch.from_numpy(x)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


class StampDataset(Dataset):
    """ZTF image stamp dataset.

    Each sample x has shape (3, H, W). arcsinh scaling is applied first
    to compress the dynamic range, then per-channel normalization.
    """
    def __init__(self, stamps: np.ndarray, labels: np.ndarray,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        x = np.arcsinh(np.nan_to_num(stamps, nan=0.0)).astype(np.float32)
        if mean is not None:
            x = (x - mean) / (std + 1e-8)
        self.x = torch.from_numpy(x)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def _split3(labels, val_frac, test_frac, seed):
    """Stratified three-way split. Returns (idx_train, idx_val, idx_test)."""
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels
    )
    # val_frac is expressed relative to the full dataset, so rescale for the trainval subset
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac_adjusted, random_state=seed, stratify=labels[idx_trainval]
    )
    return idx_train, idx_val, idx_test


def make_lc_loaders(
    path: str, val_frac: float = 0.15, test_frac: float = 0.15,
    batch_size: int = 32, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load and split the light curve .npz file into train/val/test DataLoaders.

    Normalization statistics are computed from training data only and applied
    consistently to val and test sets.
    """
    data = np.load(path, allow_pickle=True)
    fluxes, masks, labels = data["fluxes"], data["masks"], data["labels"]
    class_names = list(data["class_names"])

    idx_tr, idx_val, idx_te = _split3(labels, val_frac, test_frac, seed)

    # Per-band statistics from training observed values only
    flux_mean = np.zeros((1, 1, 2), dtype=np.float32)
    flux_std  = np.ones((1, 1, 2), dtype=np.float32)
    for b in range(2):
        obs = fluxes[idx_tr, :, b][masks[idx_tr, :, b] == 1]
        flux_mean[0, 0, b] = np.nanmean(obs)
        flux_std[0, 0, b]  = np.nanstd(obs)

    def _ds(idx): return LCDataset(fluxes[idx], masks[idx], labels[idx], flux_mean, flux_std)

    return (
        DataLoader(_ds(idx_tr),  batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(_ds(idx_val), batch_size=batch_size, shuffle=False),
        DataLoader(_ds(idx_te),  batch_size=batch_size, shuffle=False),
        class_names,
    )


def make_stamp_loaders(
    path: str, val_frac: float = 0.15, test_frac: float = 0.15,
    batch_size: int = 32, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load and split the stamp .npz file into train/val/test DataLoaders.

    Normalization statistics are computed from training data only and applied
    consistently to val and test sets.
    """
    data = np.load(path, allow_pickle=True)
    stamps, labels = data["stamps"], data["labels"]
    class_names = list(data["class_names"])

    idx_tr, idx_val, idx_te = _split3(labels, val_frac, test_frac, seed)

    # Per-channel normalization on arcsinh-transformed training stamps
    tr_arcsinh = np.arcsinh(np.nan_to_num(stamps[idx_tr], nan=0.0))
    mean = tr_arcsinh.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std  = tr_arcsinh.std(axis=(0, 2, 3),  keepdims=True).astype(np.float32)

    def _ds(idx): return StampDataset(stamps[idx], labels[idx], mean, std)

    return (
        DataLoader(_ds(idx_tr),  batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(_ds(idx_val), batch_size=batch_size, shuffle=False),
        DataLoader(_ds(idx_te),  batch_size=batch_size, shuffle=False),
        class_names,
    )
