"""Light curve download and preprocessing utilities for the AstroAI tutorial.

Interpolation approach follows Pérez-Carrasco et al. 2023
(https://iopscience.iop.org/article/10.3847/1538-3881/ace0c1):
  - Time relative to first detection (dt = mjd - first_mjd)
  - Non-detections before trigger included to anchor pre-explosion baseline
  - Linear interpolation onto a regular 3-day grid → fixed-length sequences
  - Unobserved time steps filled with 0 and flagged via a binary mask
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

EVAL_DATE: int = 150   # days after first detection
MIN_TIME: int  = -30   # days before first detection (pre-explosion baseline)
STEP: int      = 3     # grid spacing in days
# Sequence length T = (EVAL_DATE - MIN_TIME) / STEP = 60


_EMPTY_NONDET = pd.DataFrame(columns=["mjd", "diffmaglim", "fid"])


def download_lc(
    oid: str, client
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Download detections and non-detections for one ZTF object.

    Returns (detections, non_detections) DataFrames sorted by mjd,
    or (None, None) if the detections download fails.
    Non-detections are optional — a 403 or empty response is handled
    gracefully and returns an empty DataFrame (interpolation still works).
    """
    try:
        det = client.query_detections(oid, format="pandas")
        det = (
            det[["mjd", "magpsf", "sigmapsf", "fid"]]
            .dropna()
            .sort_values("mjd")
            .reset_index(drop=True)
        )
    except Exception as exc:
        print(f"    [SKIP] {oid}: {exc}")
        return None, None

    try:
        nondet = client.query_non_detections(oid, format="pandas")
        if len(nondet) > 0:
            nondet = (
                nondet[["mjd", "diffmaglim", "fid"]]
                .dropna()
                .sort_values("mjd")
                .reset_index(drop=True)
            )
        else:
            nondet = _EMPTY_NONDET
    except Exception:
        # Non-detections unavailable (e.g. 403 rate limit); continue without them.
        nondet = _EMPTY_NONDET

    return det, nondet


def passes_min_detections(det: pd.DataFrame, min_det_per_band: int) -> bool:
    """Return True if the object has at least min_det_per_band detections in each band."""
    return all((det["fid"] == fid).sum() >= min_det_per_band for fid in [1, 2])


def interpolate_lc(
    det: pd.DataFrame,
    nondet: pd.DataFrame | None,
    eval_date: int = EVAL_DATE,
    min_time: int = MIN_TIME,
    step: int = STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a ZTF light curve onto a regular time grid.

    Parameters
    ----------
    det    : detections DataFrame with columns [mjd, magpsf, sigmapsf, fid]
    nondet : non-detections DataFrame with columns [mjd, diffmaglim, fid]
    eval_date, min_time, step : grid parameters (days)

    Returns
    -------
    fluxes : float32 (T, 2)  interpolated magnitudes  [g-band, r-band]
    masks  : float32 (T, 2)  1 where within observed range, 0 elsewhere
    times  : float32 (T,)    uniform time axis starting at 0, spacing = step
    """
    first_mjd = det["mjd"].min()
    det = det.copy()
    det["dt"] = det["mjd"] - first_mjd

    tgrid = np.arange(min_time, eval_date, step=step, dtype=np.float32)
    times = np.arange(0, eval_date - min_time, step=step, dtype=np.float32)

    fluxes = []
    for fid in [1, 2]:
        band_mask = det["fid"] == fid
        t   = det.loc[band_mask, "dt"].values.astype(np.float32)
        mag = det.loc[band_mask, "magpsf"].values.astype(np.float32)

        # Prepend non-detections (upper limits) before trigger as baseline
        if nondet is not None and len(nondet) > 0:
            nd_band = nondet[nondet["fid"] == fid].copy()
            nd_band["dt"] = (nd_band["mjd"] - first_mjd).astype(np.float32)
            nd_band = nd_band[nd_band["dt"] < 0]
            if len(nd_band) > 0:
                t   = np.concatenate([nd_band["dt"].values, t])
                mag = np.concatenate([nd_band["diffmaglim"].values.astype(np.float32), mag])
                order = np.argsort(t)
                t, mag = t[order], mag[order]

        in_range = (t >= min_time) & (t <= eval_date)
        t, mag = t[in_range], mag[in_range]

        # Deduplicate by time: duplicate MJDs cause slope = (y-y)/0 = NaN in interp1d
        _, unique_idx = np.unique(t, return_index=True)
        t, mag = t[unique_idx], mag[unique_idx]

        f = interp1d(t, mag, kind="linear", bounds_error=False, fill_value=0.0)
        band_flux = f(tgrid).astype(np.float32)

        # Guard: replace any residual NaN (e.g. single-point bands) with 0
        band_flux = np.nan_to_num(band_flux, nan=0.0)
        fluxes.append(band_flux)

    fluxes = np.stack(fluxes, axis=-1)          # (T, 2)
    masks  = (fluxes != 0).astype(np.float32)   # (T, 2)
    return fluxes, masks, times
